import os
import sys
import numpy as np
import netCDF4
import xarray as xr 

from regrid_gases import regrid_gases

def assign_hybrid_coordinates_and_flip(file_name):
    """
    Creates hybrid sigma coordinates for height.
    -- Coordinates for level centers ("level")
    -- Coordinates for level interfaces (half levels or "hlevel")

    Flips the data such that level=0 is at the top of the atmosphere (TOA)
    Layer numbering increases from TOA toward the surface
    """
    data = netCDF4.Dataset(file_name, "a")

    hlevel = data.dimensions['hlevel'].size
    level = data.dimensions['level'].size

    ilev = data.createVariable("hlevel","f8",("hlevel"))
    ilev.long_name = "hybrid level at layer interfaces"	
    ilev.standard_name = "hybrid_sigma_pressure"
    ilev.units = "level"
    ilev.positive = "down" 
    ilev.formula = "hyai hybi (ilev=hyai+hybi*aps)" 
    ilev.formula_terms = "ap: hyai b: hybi ps: aps" 
    for i in np.arange(hlevel):
        ilev[i] = i+1

    #mlev = data.createVariable("level","f8",("level"))
    mlev = data.variables["level"]
    mlev.long_name = "hybrid level at layer midpoints"	
    mlev.standard_name = "hybrid_sigma_pressure"
    mlev.units = "level"
    mlev.positive = "down" 
    mlev.formula = "hyam hybm (mlev=hyam+hybm*aps)" 
    mlev.formula_terms = "ap: hyam b: hybm ps: aps" 

    for i in np.arange(level):
        mlev[i] = i+1

    hyai = data.createVariable("hyai","f8",("hlevel"))
    hyai.long_name = "hybrid A coefficient at layer interfaces"
    hyai.units = 'Pa'

    hybi = data.createVariable("hybi","f8",("hlevel")) 
    hybi.long_name = "hybrid B coefficient at layer interfaces"
    hybi.units = 'Pa'

    hyam = data.createVariable("hyam","f8",("level"))
    hyam.long_name = "hybrid A coefficient at layer midpoints"
    hyam.units = 'Pa'

    hybm = data.createVariable("hybm","f8",("level")) 
    hybm.long_name = "hybrid B coefficient at layer midpoints"
    hybm.units = 'Pa'

    ap = data.variables['ap']
    bp = data.variables['bp']

    # check if data is flipped
    if bp[0] < bp[-1]:
        print(f"Error: Data in {file_name} is not vertically flipped")
        print(f"bp[0] = {bp[0]}, bp[-1] = {bp[-1]}")
        sys.exit(1)

    ap = np.flip(ap,axis=0)
    bp = np.flip(bp,axis=0)

    hyai[:] = ap[:]
    hybi[:] = bp[:]

    hyam[:] = 0.5 * (ap[1:] + ap[:-1])
    hybm[:] = 0.5 * (bp[1:] + bp[:-1])

    sp_v = data.variables['Psurf']
    sp_v.standard_name = "Dry air surface pressure"
    n2o_dat = data.variables['N2O'][:].data
    n2o_dat = np.flip(n2o_dat,axis=1)
    data.variables['N2O'][:] = n2o_dat

    data.close()

def reconcile_time_samples(file_name_1, file_name_2, output_name_1, output_name_2):
    """
    Given two Datasets, each dataset is reduced to the sampling times common to both original sets
    """
    dt_1 = xr.open_dataset(file_name_1, engine='netcdf4')
    dt_2 = xr.open_dataset(file_name_2, engine='netcdf4')

    time_1 = dt_1.coords['time'].values
    time_2 = dt_2.coords['time'].values
  
    time_common = []
    for t_1 in time_1:
        if t_1 in time_2:
            time_common.append(t_1)

    sampled_dt_1 = dt_1.sel(time=time_common)
    sampled_dt_2 = dt_2.sel(time=time_common)

    xr.Dataset.to_netcdf(sampled_dt_1, output_name_1)
    xr.Dataset.to_netcdf(sampled_dt_2, output_name_2)

def regrid_n2o(original_data_dir, processed_data_dir, mode, year):

    months = [str(m).zfill(2) for m in range(1,13)]
    for month in months[:1]:

        input_directory=f'{original_data_dir}{mode}/{year}/{month}/'

        file_name = f"{input_directory}CAMS_n2o_{year}-{month}.tar.latest.gz"

        output_directory=f'{processed_data_dir}{mode}/{year}/{month}/'

        cmd = f'tar -xvzf {file_name} -C {output_directory}'
        os.system(cmd)

        cmd = f'cp {output_directory}cams73_latest_n2o_conc_surface_inst_{year}{month}.nc {output_directory}tmp.n2o.1.nc'
        os.system(cmd)

        assign_hybrid_coordinates_and_flip(f'{output_directory}tmp.n2o.1.nc')

        cmd = f'ncks -O -x -v ap,bp {output_directory}tmp.n2o.1.nc {output_directory}tmp.n2o.2.nc'
        os.system(cmd)

        cmd = f'cdo remapbil,{original_data_dir}../icon_grid_0009_R02B03_R.nc {output_directory}tmp.n2o.2.nc {output_directory}tmp.n2o.3.nc'
        os.system(cmd)

        reconcile_time_samples(f'{output_directory}CAMS_{year}-{month}.2.nc', f'{output_directory}tmp.n2o.3.nc', f'{output_directory}CAMS_{year}-{month}.3.nc', f'{output_directory}tmp.n2o.4.nc')

        print("time samples reconciled")

        cmd = f'ncks -A -v sp {output_directory}CAMS_{year}-{month}.3.nc {output_directory}tmp.n2o.4.nc'
        os.system(cmd)

        print("copied from n2o to gases")

        cmd = f'cdo remapeta,{original_data_dir}../newvct {output_directory}tmp.n2o.4.nc {output_directory}tmp.n2o.5.nc'
        os.system(cmd)

        cmd = f'ncrename -d lev,layer {output_directory}tmp.n2o.5.nc'
        os.system(cmd)

        cmd = f'ncks -A -v N2O {output_directory}tmp.n2o.5.nc {output_directory}CAMS_{year}-{month}.3.nc'
        os.system(cmd)


def evaluate_clat_bnds(cell, file_name_1, file_name_2):
    """
    Given two Datasets, each dataset is reduced to the sampling times common to both original sets
    """
    dt_1 = xr.open_dataset(file_name_1, engine='netcdf4')
    dt_2 = xr.open_dataset(file_name_2, engine='netcdf4')

    clat_bnds_1 = dt_1['clat_bnds'].values
    clat_bnds_2 = dt_2['clat_bnds'].values

    print(f"clat 1 = {clat_bnds_1[cell,:]}")
    print(f"clat 2 = {clat_bnds_2[cell,:]}")

    clon_bnds_1 = dt_1['clon_bnds'].values
    clon_bnds_2 = dt_2['clon_bnds'].values

    print(f"clon 1 = {clon_bnds_1[cell,:]}")
    print(f"clon 2 = {clon_bnds_2[cell,:]}")

if __name__ == "__main__":
    original_data_dir = '/data-T1/hws/CAMS/original_data/'
    processed_data_dir = '/data-T1/hws/CAMS/processed_data/'

    if True:
        if len(sys.argv) != 3:
            print("Usage: regrid_n2o mode year")
            print("Mode must be one of the following: 'training', 'testing', or 'cross_validation'")
            sys.exit(1)
        elif sys.argv[1] not in ['training','testing','cross_validation']:
            print("Second argument must be one of the following: 'training', 'testing', or 'cross_validation'")
            sys.exit(1)

        mode = sys.argv[1]
        year = sys.argv[2]
    else:
        mode = "training"
        year = "2008"
        month = '01'
        output_directory=f'{processed_data_dir}{mode}/{year}/{month}/'

    regrid_gases('/data-T1/hws/CAMS/',mode, year)
    regrid_n2o(original_data_dir, processed_data_dir, mode, year)
    #evaluate_clat_bnds(130, f"{output_directory}CAMS_{year}-{month}.final.2.nc",
    #f"{output_directory}tmp.n2o.4.nc")

