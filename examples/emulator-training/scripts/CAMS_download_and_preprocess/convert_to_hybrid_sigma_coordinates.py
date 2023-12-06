# Converts ap,bp to hybrid sigma coordinate coefficients
import sys
#import xarray
import netCDF4 #pylint: disable=no-name-in-module
#from netCDF4 import netCDF4.Dataset
import numpy as np
#import glob

#data_directory = "/data-T1/hws/CAMS/original_data/n2o/"
#file_in = "cams73_latest_n2o_conc_surface_inst_201502.nc"
#file_in = "tmp.0.nc"
#file_out = "VERTICAL_Ref.nc"

# no longer used
def assign_hybrid_coordinates_old(file_in, file_out):
    """
    Following ECHAM standard in example ECHAM file:
    test_echam_spectral.nc 
    on CF standards site:
    https://www.unidata.ucar.edu/software/netcdf/examples/files.html

    Note that level=0 is at the top of the atmosphere (TOA)
    Layers increase toward the surface
    """
    data_in = netCDF4.Dataset(file_in, "r")
    data_out = netCDF4.Dataset(file_out, "w")

    n_i = data_in.dimensions['hlevel'].size
    n_m = data_in.dimensions['level'].size

    data_out.createDimension('mlev',n_m)
    data_out.createDimension('ilev',n_i)

    ilev = data_out.createVariable("ilev","f8",("ilev"))
    ilev.long_name = "hybrid level at layer interfaces"	
    ilev.standard_name = "hybrid_sigma_pressure"
    ilev.units = "level"
    ilev.positive = "down" 
    ilev.formula = "hyai hybi (ilev=hyai+hybi*aps)" 
    ilev.formula_terms = "ap: hyai b: hybi ps: aps" 
    for i in np.arange(n_i):
        ilev[i] = i+1

    mlev = data_out.createVariable("mlev","f8",("mlev"))
    mlev.long_name = "hybrid level at layer midpoints"	
    mlev.standard_name = "hybrid_sigma_pressure"
    mlev.units = "level"
    mlev.positive = "down" 
    mlev.formula = "hyam hybm (mlev=hyam+hybm*aps)" 
    mlev.formula_terms = "ap: hyam b: hybm ps: aps" 
    mlev.borders = "ilev"
    for i in np.arange(n_m):
        mlev[i] = i+1

    hyai = data_out.createVariable("hyai","f8",("ilev"))
    hyai.long_name = "hybrid A coefficient at layer interfaces"
    hyai.units = 'Pa'

    hybi = data_out.createVariable("hybi","f8",("ilev")) 
    hybi.long_name = "hybrid B coefficient at layer interfaces"
    hybi.units = 'Pa'

    hyam = data_out.createVariable("hyam","f8",("mlev"))
    hyam.long_name = "hybrid A coefficient at layer midpoints"
    hyam.units = 'Pa'

    hybm = data_out.createVariable("hybm","f8",("mlev")) 
    hybm.long_name = "hybrid B coefficient at layer midpoints"
    hybm.units = 'Pa'

    ap = data_in.variables['ap']
    bp = data_in.variables['bp']

    ap = np.flip(ap,axis=0)
    bp = np.flip(bp,axis=0)

    hyai[:] = ap[:]
    hybi[:] = bp[:]

    hyam[:] = 0.5 * (ap[1:] + ap[:-1])
    hybm[:] = 0.5 * (bp[1:] + bp[:-1])

    data_out.close()
    data_in.close()


def assign_hybrid_coordinates_and_flip(file_name):
    """
    Following ECHAM standard in example ECHAM file:
    test_echam_spectral.nc 
    on CF standards site:
    https://www.unidata.ucar.edu/software/netcdf/examples/files.html

    Note that level=0 is at the top of the atmosphere (TOA)
    Layers increase toward the surface
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


if __name__ == "__main__":
    # Input data file
    if len(sys.argv) != 2:
        print("usage: preproc_n20 [file_name.nc]")
        print(f"Number of command line args = {len(sys.argv)}")
        sys.exit(1)
    else:
        file_name = sys.argv[1]
        assign_hybrid_coordinates_and_flip(file_name)





