# Converts ap,bp to hybrid sigma coordinate coefficients
import os
from netCDF4 import Dataset
import numpy as np
import glob

data_directory = "/data-T1/hws/CAMS/original_data/n2o/"
file_in = "cams73_latest_n2o_conc_surface_inst_201502.nc"
#file_in = "tmp.0.nc"
file_out = "VERTICAL_Ref.nc"

def assign_hybrid_coordinates():
    """
    Following ECHAM standard in example ECHAM file:
    test_echam_spectral.nc 
    on CF standards site:
    https://www.unidata.ucar.edu/software/netcdf/examples/files.html

    Note that level=0 is at the top of the atmosphere (TOA)
    Layers increase toward the surface
    """
    data_in = Dataset(data_directory + file_in, "r")
    data_out = Dataset(data_directory + file_out, "w")

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

def flip_coordinates():
    file_name = "tmp.3.nc"
    data = Dataset(data_directory + file_name,'a')
    sp_v = data.variables['aps']
    sp_v.standard_name = "surface_air_pressure"
    n2o_dat = data.variables['N2O'][:].data
    n2o_dat = np.flip(n2o_dat,axis=1)
    data.variables['N2O'][:] = n2o_dat
    data.close()

def test_glob():
    l1 = glob.glob(data_directory + '*')
    print(l1)

if __name__ == "__main__":
    #test_glob()
    #assign_hybrid_coordinates()
    flip_coordinates()




