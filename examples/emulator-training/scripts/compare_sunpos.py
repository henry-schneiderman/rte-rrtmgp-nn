from netCDF4 import Dataset
import numpy as np
import xarray as xr

data_dir       = "/data-T1/hws/tmp/"
#file_name_in   = data_dir + "CAMS_2014_RFMIPstyle.nc"
#file_name_in   = data_dir + "CAMS_2009-2018_sans_2014-2015_RFMIPstyle.nc"
#file_name_out  = data_dir + "RADSCHEME_data_g224_CAMS_2014.nc"
#file_name_out  = data_dir + "RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.nc"
#file_name_out2  = data_dir + "RADSCHEME_data_g224_CAMS_2014.2.nc"

file_name_original = data_dir + "RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"

file_name = "/data-T1/hws/CAMS/original_data/n2o/tmp.7.nc"
n = 19
t = 13

data_1 = xr.open_dataset(file_name) 
data_original = xr.open_dataset(file_name_original) 

eps = 1.0e-08
sun_1 = data_1.variables["solar_zenith_angle"].data
sun_2 = data_1.variables["solar_zenith_angle_2"].data
mu_orig = data_original.variables["mu0"].data

t = data_1.variables["time"]
if True:

    #print(f"Full P0[0,:,{n}] = {co2_2[t,n,:]}\n")
    #print(f"Half P1[0,:,{n}] = {co2_1[t,:,n]}\n")

    print(f"time 0-4 = {t[:4]}")
    print(f"time 4-8 = {t[4:8]}\n")
    print(f"time 8-12 = {t[8:12]}\n")
    print(f"time 12-16 = {t[12:16]}\n")

    print(f"sun_1 = {np.cos(np.pi * sun_1[:4,n] / 180.0)}")
    print(f"sun_2 = {np.cos(np.pi * sun_2[:4,n] / 180.0)}")
    print(f"Original mu = {mu_orig[:4,n]}\n")

    print(f"sun_1 = {np.cos(np.pi * sun_1[4:8,n] / 180.0)}")
    print(f"sun_2 = {np.cos(np.pi * sun_2[4:8,n] / 180.0)}")
    print(f"Original mu = {mu_orig[4:8,n]}\n")

    print(f"sun_1 = {np.cos(np.pi * sun_1[8:12,n] / 180.0)}")
    print(f"sun_2 = {np.cos(np.pi * sun_2[8:12,n] / 180.0)}")
    print(f"Original mu = {mu_orig[8:12,n]}\n")

    print(f"sun_1 = {np.cos(np.pi * sun_1[12:16,n] / 180.0)}")
    print(f"sun_2 = {np.cos(np.pi * sun_2[12:16,n] / 180.0)}")
    print(f"Original mu = {mu_orig[12:16,n]}\n")

    print(f"sun_1 = {np.cos(np.pi * sun_1[4,23:43] / 180.0)}")

    
    #print(f"no2 ratio should be constant = {no2_2[t,n,:] / n2o_1[t,:,n] }")