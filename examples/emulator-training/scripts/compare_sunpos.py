from netCDF4 import Dataset
import numpy as np
import xarray as xr

data_dir       = "/home/hws/tmp/"
#file_name_in   = data_dir + "CAMS_2014_RFMIPstyle.nc"
#file_name_in   = data_dir + "CAMS_2009-2018_sans_2014-2015_RFMIPstyle.nc"
#file_name_out  = data_dir + "RADSCHEME_data_g224_CAMS_2014.nc"
#file_name_out  = data_dir + "RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.nc"
#file_name_out2  = data_dir + "RADSCHEME_data_g224_CAMS_2014.2.nc"



file_name = "/data-T1/hws/CAMS/original_data/n2o/tmp.7.nc"
n = 19
t = 13

data_1 = xr.open_dataset(file_name) 

eps = 1.0e-08
sun_1 = data_1.variables["solar_zenith_angle"]
sun_2 = data_1.variables["solar_zenith_angle_2"]
t = data_1.variables["time"]
if True:

    #print(f"Full P0[0,:,{n}] = {co2_2[t,n,:]}\n")
    #print(f"Half P1[0,:,{n}] = {co2_1[t,:,n]}\n")

    print(f"time 0-4 = {t[:4]}")
    print(f"time 4-8 = {t[4:8]}\n")
    print(f"time 8-12 = {t[8:12]}\n")
    print(f"time 12-16 = {t[12:16]}\n")

    print(f"sun_1 = {sun_1[:4,n]}")
    print(f"sun_2 = {sun_2[:4,n]}\n")

    print(f"sun_1 = {sun_1[4:8,n]}")
    print(f"sun_2 = {sun_2[4:8,n]}\n")

    print(f"sun_1 = {sun_1[8:12,n]}")
    print(f"sun_2 = {sun_2[8:12,n]}\n")

    print(f"sun_1 = {sun_1[12:16,n]}")
    print(f"sun_2 = {sun_2[12:16,n]}\n")

    
    #print(f"no2 ratio should be constant = {no2_2[t,n,:] / n2o_1[t,:,n] }")