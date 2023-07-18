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

rsd_direct = data.variables["rsd_dir"]
rsd = data.variables["rsd"]
rsu = data.variables["rsu"]
mu = data.variables["mu0"]
pres = data.variables["pres_level"]
lwp = data.variables["cloud_lwp"]
iwp = data.variables["cloud_iwp"]
h20 = data.variables['rrtmgp_sw_input'][:,:,:,2]
o2 = data.variables['rrtmgp_sw_input'][:,:,:,3]
clwp = data.variables['cloud_lwp']
ciwp = data.variables['cloud_iwp']

n = rsd.data / rsd[:,:,0:1].data
print(f"rsd mean = {np.mean(n,axis=(0,1))}")
print ("")
n = rsd_direct.data / rsd_direct[:,:,0:1].data
print(f"rsd_direct mean = {np.mean(n,axis=(0,1))}")
print ("")
print("T-direct 1,200,41:61 = " + str(rsd_direct[1,200,41:61].data / (rsd_direct[1,200,40:60].data + 0.00001)))
print (" ")

print("T-direct 19,340,51:61 = " + str(rsd_direct[19,340,51:61].data / (rsd_direct[19,340,50:60].data + 0.00001)))
print (" ")


print("T-direct 60,540,41:61 = " + str(rsd_direct[60,540,41:61].data / (rsd_direct[60,540,40:60].data + 0.00001)))
print (" ")

print("h20 60,540,41:61 = " + str(h20[60,540,41:61].data))
print (" ")

print("clwp 60,540,41:61 = " + str(clwp[60,540,41:61].data))
print (" ")

print("ciwp 60,540,41:61 = " + str(ciwp[60,540,41:61].data))
print (" ")


print (" ")

total = np.sum(h20[60,:,0:40].data, axis=(0,))
print(f'Sum(h20[60,:,0:40]) = {total} ')
print (" ")

total = np.sum(o2[60,:,0:40].data, axis=(0,))
print(f'Sum(o2[60,:,0:40]) = {total} ')
print (" ")

print (" ")

total = np.sum(clwp[60,:,20:40].data, axis=(0,))
print(f'Sum(clwp[60,:,20:40]) = {total} ')
print (" ")

total = np.sum(ciwp[60,:,20:40].data, axis=(0,))
print(f'Sum(ciwp[60,:,20:40]) = {total} ')
print (" ")

print("T-direct 60,900,20:40 = " + str(np.sum(rsd_direct[60,:,21:41].data, axis = (0,)) / (np.sum(rsd_direct[60,:,20:40].data, axis=(0,)) + 0.00001)))
print (" ")

print("clwp 60,900,0:10 = " + str(clwp[60,900,0:10].data))
print (" ")

print("ciwp 60,900,0:10 = " + str(ciwp[60,900,0:10].data))
print (" ")

print("mu  = " + str(mu[0:7,200].data))
print("412.0 * mu  = " + str(1412.0 * mu[0:7,200].data))
print("rsd = " + str(rsd[0:7,200,0].data))
print("rsd_direct = " + str(rsd_direct[0:7,200,0].data))
print (" ")

print("rsd - 55 = " + str(rsd[0:7,200,55].data))
print("rsd_direct - 55 = " + str(rsd_direct[0:7,200,55].data))
print (" ")
print("rsd - 0:60:10 = " + str(rsd[1,200,0:60:10].data))
print("rsd_direct - 0:60:10 = " + str(rsd_direct[1,200,0:60:10].data))
print (" ")
print("rsd - 40:50 = " + str(rsd[1,200,41:51].data))
print("rsd_direct - 40:50 = " + str(rsd_direct[1,200,41:51].data))
print (" ")
print("lwp 40:50 = " + str(lwp[1,200,40:50].data))
print("iwp 40:50 = " + str(iwp[1,200,40:50].data))
print (" ")
print (" ")
print("rsd - 5,0:60:10 = " + str(rsd[5,200,0:60:10].data))
print("rsd_direct - 5,0:60:10 = " + str(rsd_direct[5,200,0:60:10].data))
print (" ")
print("rsd - 5,0:60:10 = " + str(rsd[5,200,31:41].data))
print("rsd_direct - 5,0:60:10 = " + str(rsd_direct[5,200,31:41].data))
print("lwp 5,40:50 = " + str(lwp[5,200,30:40].data))
print("iwp 5,40:50 = " + str(iwp[5,200,30:40].data))
print (" ")
print("mu  = " + str(mu[7,200].data))
print("rsd  = " + str(1412.0 * mu[7,200].data))
print("rsd = " + str(rsd[7,200, 10:20].data))

print (" ")
print("mu  = " + str(mu[0:7,201].data))
print("rsd = " + str(rsd[0:7,201,0].data))
print("rsd = " + str(rsd[0,201,0:4].data))
print (" ")
print("mu  = " + str(mu[0:7,401].data))
print("rsd = " + str(rsd[0:7,401,0].data))

print(" ")
print(" ")

print("rsu top = " + str(rsu[7,200,0:5].data))
print("rsu bottom = " + str(rsu[7,200,-5:].data))
print("rsu tops = " + str(rsu[7,200:209,0].data))
print("rsu bottoms = " + str(rsu[7,200:209,-1].data))
print(" ")
print(str(pres[0,200,:].data))
print(str(pres[0,201,:].data))



