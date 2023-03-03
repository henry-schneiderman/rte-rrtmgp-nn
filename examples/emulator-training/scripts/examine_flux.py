from netCDF4 import Dataset
import numpy as np
import xarray as xr

data_dir       = "/home/hws/tmp/"
#file_name_in   = data_dir + "CAMS_2014_RFMIPstyle.nc"
#file_name_in   = data_dir + "CAMS_2009-2018_sans_2014-2015_RFMIPstyle.nc"
#file_name_out  = data_dir + "RADSCHEME_data_g224_CAMS_2014.nc"
#file_name_out  = data_dir + "RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.nc"
#file_name_out2  = data_dir + "RADSCHEME_data_g224_CAMS_2014.2.nc"
file_name_out2  = data_dir + "RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"
data = xr.open_dataset(file_name_out2) #Dataset(file_name_in)


rsd = data.variables["rsd"]
mu = data.variables["mu0"]
pres = data.variables["pres_level"]

print("mu  = " + str(mu[0:7,200].data))
print("rsd  = " + str(1412.0 * mu[0:7,200].data))
print("rsd = " + str(rsd[0:7,200,0].data))

print (" ")
print("mu  = " + str(mu[0:7,201].data))
print("rsd = " + str(rsd[0:7,201,0].data))
print("rsd = " + str(rsd[0,201,0:4].data))
print (" ")
print("mu  = " + str(mu[0:7,401].data))
print("rsd = " + str(rsd[0:7,401,0].data))

print(" ")
print(" ")

print(str(pres[0,200,:].data))
print(str(pres[0,201,:].data))


