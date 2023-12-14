from netCDF4 import Dataset
import numpy as np
import xarray as xr
import math

data_dir       = "/data-T1/hws/tmp/"
#file_name_in   = data_dir + "CAMS_2014_RFMIPstyle.nc"
file_name_in   = data_dir + "CAMS_2015_RFMIPstyle.nc"
#file_name_in   = data_dir + "CAMS_2009-2018_sans_2014-2015_RFMIPstyle.nc"
#file_name_out  = data_dir + "RADSCHEME_data_g224_CAMS_2014.nc"
#file_name_out  = data_dir + "RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.nc"
#file_name_out2  = data_dir + "RADSCHEME_data_g224_CAMS_2014.2.nc"
file_name_out2  = data_dir + "RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"
#file_name_out2  = data_dir + "RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"
data_1 = xr.open_dataset(file_name_in)
data_2 = xr.open_dataset(file_name_out2) #Dataset(file_name_in)

if True:
    angle = data_1.variables["solar_zenith_angle"]
    print(f'theta = {angle[0:7,200].data}')
    print(f'cos(theta) = {np.cos((np.pi / 180.0) * angle[0:7,200].data)}')
    #print(f'theta = {angle[7,2200:2204].data}')
    #print(f'cos(theta) = {np.cos((np.pi / 180.0) * angle[7,2200:2204].data)}')
    mu = data_2.variables["mu0"]
    print("mu  = " + str(mu[0:7,200].data))
    #print("mu  = " + str(mu[7,2200:2204].data))
    print("")

    c_0 = data_1.variables["nitrous_oxide"]
    c_1 = data_1.variables["nitrogen_dioxide"]
    c_h2o = data_1.variables["water_vapor"]
    c_ch4 = data_1.variables["methane"]
    c_2 = data_2.variables["rrtmgp_sw_input"]

    print("nitrous oxide  = " + str(c_0[7,2200,0:60].data))
    print (" ")
    print("nitrogen dioxide  = " + str(c_1[7,2200,0:60].data))
    print (" ")
    print("c2  = " + str(c_2[7,2200,0:60, 5].data))

    print (" ")
    print("h2o  = " + str(c_h2o[7,2200,0:6].data))
    print("h2o  = " + str(c_2[7,2200,0:6, 2].data))

    print (" ")
    print("ch4  = " + str(c_ch4[7,2200,0:6].data))
    print("ch4  = " + str(c_2[7,2200,0:6, 6].data))
else:

    c_1 = data_1.variables["clwc"]
    c_2 = data_2.variables["cloud_lwp"]


    print("c1  = " + str(c_1[7,2200,0:60].data))
    print (" ")
    print("c2  = " + str(c_2[7,2200,0:60].data))

    diff = c_1[7,2200,0:60].data / (c_2[7,2200,0:60].data + 0.00000000001)

    print (" ")
    print ("diff = " + str(diff))

    w_1 = data_1.variables["water_vapor"]
    w_2 = data_2.variables["rrtmgp_sw_input"]

    print("w1  = " + str(w_1[7,2200,0:60].data))
    print (" ")
    print("w2  = " + str(w_2[7,2200,0:60,2].data))



