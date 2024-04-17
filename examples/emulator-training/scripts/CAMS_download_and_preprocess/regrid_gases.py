import os
import sys
# run with: conda activate netcdf3.8
from netCDF4 import Dataset,num2date
import numpy as np
import xarray as xr
from sunposition import sunpos

def add_solar_zenith(file_name):
    """
    Created on Wed Sep 15 17:00:33 2021

    @author: peter

    Modified by Henry Schneiderman Oct. 12, 2023
    """
    dt = Dataset(file_name,'a')

    p   = dt.variables['pres_level'][:,:,:].data  
    lon = np.rad2deg(dt.variables['clon'][:].data)
    lat = np.rad2deg(dt.variables['clat'][:].data)

    ntime = p.shape[0]
    nsite = p.shape[2]

    # save solar zenith angle

    t_unit =  dt.variables['time'].units + '-00:00'
    t_cal  =  dt.variables['time'].calendar

    lonn = lon.reshape(1,nsite).repeat(ntime,axis=0)
    latt = lat.reshape(1,nsite).repeat(ntime,axis=0)
    lonn = lonn.reshape(nsite*ntime); 
    latt = latt.reshape(nsite*ntime)

    timedatt = dt.variables['time'][:].reshape(ntime,1).repeat(nsite,axis=1)
    timedatt = timedatt.reshape(nsite*ntime)

    t_unit =  dt.variables['time'].units 

    times = num2date(timedatt,units = t_unit,calendar = t_cal).data

    az,zen = sunpos(times.data,latt,lonn,0)[:2] #discard RA, dec, H

    sza_new = zen.reshape(ntime,nsite)
    sza = dt.createVariable("solar_zenith_angle","f4",("time", "cell"))
    sza[:] = sza_new[:]
    sza.setncattr("units", "degrees")

    dt.close()

def reconcile_time_samples_to_2nd_file(file_name_1, file_name_2, output_name_1):
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

    if len(time_common) != len(time_2):
        print(f"Times are incompatible! {file_name_1} and {file_name_2}")
        sys.exit(1)

    xr.Dataset.to_netcdf(sampled_dt_1, output_name_1)

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


def regrid_cloud_fraction(base_directory,mode,month,year):

    dir_1=f'{base_directory}original_data/{mode}/{year}/{month}/'
    dir_2=f'{base_directory}processed_data/{mode}/{year}/{month}/'

    if False:
        ## Merge into single file for each data set for each month
        print (f"{dir_1}/era5_ml_{year}-{month}-??.grb", flush=True)
        cmd = f'cdo mergetime {dir_1}/era5_ml_{year}-{month}-??.grb {dir_2}/era5_ml_{year}-{month}.grb'
        os.system(cmd)
        print("Completed mergetime")


        ## Convert to netCDF files
        cmd = f'cdo --eccodes -f nc copy {dir_2}/era5_ml_{year}-{month}.grb {dir_2}/era5_ml_{year}-{month}.nc'
        os.system(cmd)
        print("Completed conversion to netcdf")

        ## Reduce spatial resolution before attempting other operations

        cmd = f'cdo remapdis,{base_directory}/icon_grid_0009_R02B03_R.nc {dir_2}/era5_ml_{year}-{month}.nc {dir_2}/era5_ml_{year}-{month}.icon.nc'
        os.system(cmd)
        print("Completed spatial resolution remap")

    if False:

        reconcile_time_samples_to_2nd_file(f'{dir_2}era5_ml_{year}-{month}.icon.nc',
                                        f'{dir_2}CAMS_{year}-{month}.final.nc',
                                            f'{dir_2}era5_ml_{year}-{month}.icon.2.nc')
        
        print("reconciled time samples")

        dt   = xr.open_dataset(f'{dir_2}era5_ml_{year}-{month}.icon.2.nc')
        dt1 = dt.rename_dims({'cell':'site'})
        # Seems to require a variable named 'site' in order for the rest to work
        dimension_site = dt1['site'].values
        site = xr.DataArray(np.arange(1,len(dimension_site)+1), dims=("site",), name="site")
        dt1['site'] = site
        dt1.to_netcdf(f'{dir_2}era5_ml_{year}-{month}.icon.3.nc')
        dt1.close()

        print("renamed cell to site")
        #cmd = f'ncrename -d cell,site {dir_2}era5_ml_{year}-{month}.icon.2.nc {dir_2}era5_ml_{year}-{month}.icon.3.nc'
        #os.system(cmd)


    if False:

        cmd = f'ncks -A -v sp {dir_2}CAMS_{year}-{month}.final.nc {dir_2}era5_ml_{year}-{month}.icon.3.nc'
        os.system(cmd)

        print("copied in surface pressure")
    if True:
        cmd = f'cdo remapeta,{base_data_dir}newvct {dir_2}era5_ml_{year}-{month}.icon.3.nc {dir_2}era5_ml_{year}-{month}.icon.4.nc'
        os.system(cmd)

        print("Remapped vertical coordinate")

        cmd = f'ncrename -d lev,layer -v lev,layer {dir_2}era5_ml_{year}-{month}.icon.4.nc'
        os.system(cmd)

        dt_input   = xr.open_dataset(f'{dir_2}era5_ml_{year}-{month}.icon.4.nc')
        dt_input['cc'] = dt_input['cc'].transpose("time", "site", "layer")
        dt_input.to_netcdf(f'{dir_2}era5_ml_{year}-{month}.icon.5.nc')
        dt_input.close()

        cmd = f'ncks -O -x -v cloud_fraction {dir_2}CAMS_{year}-{month}.final.2.nc {dir_2}CAMS_{year}-{month}.final.3.nc'
        os.system(cmd)

        cmd = f'ncks -A -v cc {dir_2}era5_ml_{year}-{month}.icon.5.nc {dir_2}CAMS_{year}-{month}.final.3.nc'
        os.system(cmd)

def regrid_gases(base_directory,mode,month,year,use_st=False):

    dir_1=f'{base_directory}original_data/{mode}/{year}/{month}/'
    dir_2=f'{base_directory}processed_data/{mode}/{year}/{month}/'

    ## Merge into single file for each data set for each month

    cmd = f'cdo mergetime {dir_1}/CAMS_eac4_ml_{year}-{month}-?????.grb {dir_2}/CAMS_eac4_ml_{year}-{month}.grb'
    os.system(cmd)
    print("Completed 1st mergetime")

    cmd = f'cdo mergetime {dir_1}/CAMS_eac4_sfc_{year}-{month}-?????.grb {dir_2}/CAMS_eac4_sfc_{year}-{month}.grb'
    os.system(cmd)

    cmd = f'cdo mergetime {dir_1}/CAMS_egg4_ml_{year}-{month}-?????.grb {dir_2}/CAMS_egg4_ml_{year}-{month}.grb'
    os.system(cmd)

    cmd = f'cdo mergetime {dir_1}/CAMS_egg4_sfc_{year}-{month}-?????.grb {dir_2}/CAMS_egg4_sfc_{year}-{month}.grb'
    os.system(cmd)

    cmd = f'cdo mergetime {dir_1}/era5_sfc_{year}-{month}-?????.grb {dir_2}/era5_sfc_{year}-{month}.grb'
    os.system(cmd)

    # Remove the following when skin temperature is included 
    # in original downloaded CAMS_egg4_sfc (rather than downloaded separately)
    if use_st: 
        cmd = f'cdo mergetime {dir_1}/CAMS_egg4_sfc_st_{year}-{month}-?????.grb {dir_2}/CAMS_egg4_sfc_st_{year}-{month}.grb'
        os.system(cmd)

        #cmd = f'cdo mergetime {dir_1}/era5_sfc_z_{year}-{month}-?????.grb {dir_2}/era5_sfc_z_{year}-{month}.grb'
        #os.system(cmd)

    print("Completed all mergetimes")

    ## Convert to netCDF files
    cmd = f'cdo --eccodes -f nc copy {dir_2}/CAMS_eac4_ml_{year}-{month}.grb {dir_2}/CAMS_eac4_ml_{year}-{month}.nc'
    os.system(cmd)
    print("Completed 1st conversion to netcdf")

    cmd = f'cdo --eccodes -f nc copy {dir_2}/CAMS_eac4_sfc_{year}-{month}.grb {dir_2}/CAMS_eac4_sfc_{year}-{month}.nc' 
    os.system(cmd)

    cmd = f'cdo --eccodes -f nc copy {dir_2}/CAMS_egg4_ml_{year}-{month}.grb {dir_2}/CAMS_egg4_ml_{year}-{month}.nc'
    os.system(cmd)

    cmd = f'cdo --eccodes -f nc copy {dir_2}/CAMS_egg4_sfc_{year}-{month}.grb {dir_2}/CAMS_egg4_sfc_{year}-{month}.nc'
    os.system(cmd)
    print ("Completed all conversions to netcdf")

    cmd = f'cdo --eccodes -f nc copy {dir_2}/era5_sfc_{year}-{month}.grb {dir_2}/era5_sfc_{year}-{month}.nc'
    os.system(cmd)

    # Remove the following when skin temperature is included 
    # in original downloaded CAMS_egg4_sfc (rather than downloaded separately)
    if use_st:

        cmd = f'cdo --eccodes -f nc copy {dir_2}/CAMS_egg4_sfc_st_{year}-{month}.grb {dir_2}/CAMS_egg4_sfc_st_{year}-{month}.nc'
        os.system(cmd)

        #cmd = f'cdo --eccodes -f nc copy {dir_2}/era5_sfc_z_{year}-{month}.grb {dir_2}/era5_sfc_z_{year}-{month}.nc'
        #os.system(cmd)

    ## Reduce spatial resolution before attempting other operations

    cmd = f'cdo remapcon,{base_directory}/icon_grid_0009_R02B03_R.nc {dir_2}/CAMS_eac4_ml_{year}-{month}.nc {dir_2}/CAMS_eac4_ml_{year}-{month}.icon.nc'
    os.system(cmd)
    print("Completed first spatial resolution remap")

    cmd = f'cdo remapcon,{base_directory}/icon_grid_0009_R02B03_R.nc {dir_2}/CAMS_eac4_sfc_{year}-{month}.nc {dir_2}/CAMS_eac4_sfc_{year}-{month}.icon.nc'
    os.system(cmd)

    cmd = f'cdo remapcon,{base_directory}/icon_grid_0009_R02B03_R.nc {dir_2}/CAMS_egg4_ml_{year}-{month}.nc {dir_2}/CAMS_egg4_ml_{year}-{month}.icon.nc'
    os.system(cmd)

    cmd = f'cdo remapcon,{base_directory}/icon_grid_0009_R02B03_R.nc {dir_2}/CAMS_egg4_sfc_{year}-{month}.nc {dir_2}/CAMS_egg4_sfc_{year}-{month}.icon.nc'
    os.system(cmd)

    cmd = f'cdo remapcon,{base_directory}/icon_grid_0009_R02B03_R.nc {dir_2}/era5_sfc_{year}-{month}.nc {dir_2}/era5_sfc_{year}-{month}.icon.nc'
    os.system(cmd)

    print("Completed all spatial resolution remaps")

    if use_st:

        cmd = f'cdo remapcon,{base_directory}/icon_grid_0009_R02B03_R.nc {dir_2}/CAMS_egg4_sfc_st_{year}-{month}.nc {dir_2}/CAMS_egg4_sfc_st_{year}-{month}.icon.nc'
        os.system(cmd)

        #cmd = f'cdo remapcon,{base_directory}/icon_grid_0009_R02B03_R.nc {dir_2}/era5_sfc_z_{year}-{month}.nc {dir_2}/era5_sfc_z_{year}-{month}.icon.nc'
        #os.system(cmd)


    ## Move relevant variables into CAMS_eac4_ml file
    cmd = f'ncks -A {dir_2}/CAMS_eac4_sfc_{year}-{month}.icon.nc {dir_2}/CAMS_eac4_ml_{year}-{month}.icon.nc'
    os.system(cmd)

    cmd = f'ncks -A -v ch4,co2 {dir_2}/CAMS_egg4_ml_{year}-{month}.icon.nc {dir_2}/CAMS_eac4_ml_{year}-{month}.icon.nc'
    os.system(cmd)

    # See https://codes.ecmwf.int/grib/param-db/
    cmd = f'ncrename -v \\2t,t2m {dir_2}/CAMS_egg4_sfc_{year}-{month}.icon.nc'
    os.system(cmd)

    cmd = f'ncks -A -v alnid,alnip,aluvd,aluvp {dir_2}/era5_sfc_{year}-{month}.icon.nc {dir_2}/CAMS_eac4_ml_{year}-{month}.icon.nc'
    os.system(cmd)

    if use_st: 
        cmd = f'ncks -A -v skt,asn,sd {dir_2}/CAMS_egg4_sfc_st_{year}-{month}.icon.nc {dir_2}/CAMS_eac4_ml_{year}-{month}.icon.nc'
        os.system(cmd)
    else:
        cmd = f'ncks -A -v skt,asn,sd {dir_2}/CAMS_egg4_sfc_{year}-{month}.icon.nc {dir_2}/CAMS_eac4_ml_{year}-{month}.icon.nc'
        os.system(cmd)

    cmd = f'ncks -A -v t2m,fal,tisr {dir_2}/CAMS_egg4_sfc_{year}-{month}.icon.nc {dir_2}/CAMS_eac4_ml_{year}-{month}.icon.nc'
    os.system(cmd)

    cmd = f'cp {dir_2}/CAMS_eac4_ml_{year}-{month}.icon.nc {dir_2}/CAMS_{year}-{month}.1.nc'
    os.system(cmd)

    # Generate pressures at interfaces between layers (i.e., at the 'levels')
    cmd = f'cdo -pressure_hl {dir_2}/CAMS_{year}-{month}.1.nc {dir_2}/CAMS_{year}-{month}.pressure_hl.1.nc'
    os.system(cmd)
    cmd = f'cdo -pressure_fl {dir_2}/CAMS_{year}-{month}.1.nc {dir_2}/CAMS_{year}-{month}.pressure_fl.1.nc'
    os.system(cmd)

    cmd = f'ncrename -d lev,level {dir_2}/CAMS_{year}-{month}.pressure_hl.1.nc'
    os.system(cmd)

    cmd = f'ncrename -d lev,layer {dir_2}/CAMS_{year}-{month}.pressure_fl.1.nc'
    os.system(cmd)

    cmd = f'ncrename -v pressure,pres_level {dir_2}/CAMS_{year}-{month}.pressure_hl.1.nc'
    os.system(cmd)

    cmd = f'ncrename -v pressure,pres_layer {dir_2}/CAMS_{year}-{month}.pressure_fl.1.nc'
    os.system(cmd)

    cmd = f'ncrename -d lev,layer {dir_2}/CAMS_{year}-{month}.1.nc'
    os.system(cmd)

    cmd = f'ncks -A -v pres_level {dir_2}/CAMS_{year}-{month}.pressure_hl.1.nc {dir_2}/CAMS_{year}-{month}.1.nc'
    os.system(cmd)

    cmd = f'ncks -A -v pres_layer {dir_2}/CAMS_{year}-{month}.pressure_fl.1.nc {dir_2}/CAMS_{year}-{month}.1.nc'
    os.system(cmd)

    cmd = f'cdo setcalendar,standard {dir_2}/CAMS_{year}-{month}.1.nc {dir_2}/CAMS_{year}-{month}.2.nc'
    os.system(cmd)

    add_solar_zenith(f'{dir_2}/CAMS_{year}-{month}.2.nc')

if __name__ == "__main__":
    base_data_dir = '/data-T1/hws/CAMS/'

    if False:
        if len(sys.argv) != 4:
            print("Usage: regrid_n2o mode year")
            print("Mode must be one of the following: 'training', 'testing', or 'cross_validation'")
            sys.exit(1)
        elif sys.argv[1] not in ['training','testing','cross_validation']:
            print("Second argument must be one of the following: 'training', 'testing', or 'cross_validation'")
            sys.exit(1)

        mode = sys.argv[1]
        month = sys.argv[2]
        year = sys.argv[3]

        regrid_gases(base_data_dir, mode, month, year, use_st=True)
    else:
        mode = "training"
        year = "2008"
        month = '01'

        regrid_cloud_fraction(base_data_dir, mode, month, year)
        if False:
            dir_2=f'{base_data_dir}processed_data/{mode}/{year}/{month}/'
            reconcile_time_samples(f'{dir_2}tmp/cw.1.nc',
                                            f'{dir_2}CAMS_{year}-{month}.final.nc',
                                                f'{dir_2}tmp/cw.2.nc',
                                                f'{dir_2}tmp/cams.nc')


    if False:
        month = '01'
        cmd = f'ncrename -v \\2t,t2m {base_data_dir}/CAMS_egg4_sfc_{year}-{month}.icon.nc'
        print(cmd)


