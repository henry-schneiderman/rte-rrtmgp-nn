from netCDF4 import Dataset
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

data_dir       = "/home/hws/tmp/"
#file_name_in   = data_dir + "CAMS_2014_RFMIPstyle.nc"
file_name_in   = data_dir + "CAMS_2009-2018_sans_2014-2015_RFMIPstyle.nc"
#file_name_out  = data_dir + "RADSCHEME_data_g224_CAMS_2014.nc"
#file_name_out  = data_dir + "RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.nc"
file_name_out3  = data_dir + "RADSCHEME_data_g224_CAMS_2014.2.nc"
file_name_out2  = data_dir + "RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"
file_name_in_2 = data_dir + "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"
data_in = xr.open_dataset(file_name_in)
data_in_2 = xr.open_dataset(file_name_in_2)
data_out = xr.open_dataset(file_name_out2) #Dataset(file_name_in)
data_out_3 = xr.open_dataset(file_name_out3)

rsd_direct = data_out.variables["rsd_dir"]
rsd = data_out.variables["rsd"]
rsu = data_out.variables["rsu"]
print(f'Average flux down = {np.mean(np.abs(rsd.data))}')
print(f'Average flux up = {np.mean(np.abs(rsu.data))}')
mu = data_out.variables["mu0"].data
mu_2 = data_in_2.variables["mu0"].data
mu_3 = data_out_3.variables["mu0"].data

fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
axs[0].hist(np.arccos(mu.flatten()),bins=40, density=True)
axs[1].hist(np.arccos(mu_3.flatten()),bins=40, density=True)
axs[2].hist(np.arccos(mu_2.flatten()),bins=40, density=True)
plt.show() 

pres_level = data_out.variables["pres_level"]
orig_pres_level = data_in.variables["pres_level"].data
orig_co2 = data_in.variables['carbon_dioxide'].data
lwp = data_out.variables["cloud_lwp"]
iwp = data_out.variables["cloud_iwp"]
h20 = data_out.variables['rrtmgp_sw_input'][:,:,:,2]
orig_h20 = data_in.variables['water_vapor'][:,:,:].data
orig_o2 = data_in.variables['oxygen_GM'][:].data
orig_n2 = data_in.variables['nitrogen_GM'][:].data
orig_level = data_in.variables['lev'][:].data
o3 = data_out.variables['rrtmgp_sw_input'][:,:,:,3]
c2o = data_out.variables['rrtmgp_sw_input'][:,:,:,4]
pres_layer = data_out.variables['rrtmgp_sw_input'][:,:,:,1]
temp_layer = data_out.variables['rrtmgp_sw_input'][:,:,:,0]
clwp = data_out.variables['cloud_lwp']
ciwp = data_out.variables['cloud_iwp']

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

total = np.sum(o3[60,:,0:40].data, axis=(0,))
print(f'Sum(o3[60,:,0:40]) = {total} ')
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
print("412.0 * mu  = " + str(1412.0 * mu[0:7,200]))
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
print("rsd  = " + str(1412.0 * mu[7,200]))
print("rsd = " + str(rsd[7,200, 10:20].data))

print (" ")
print("mu  = " + str(mu[0:7,201]))
print("rsd = " + str(rsd[0:7,201,0].data))
print("rsd = " + str(rsd[0,201,0:4].data))
print (" ")
print("mu  = " + str(mu[0:7,401]))
print("rsd = " + str(rsd[0:7,401,0].data))

print(" ")
print(" ")

print("rsu top = " + str(rsu[7,200,0:5].data))
print("rsu bottom = " + str(rsu[7,200,-5:].data))
print("rsu tops = " + str(rsu[7,200:209,0].data))
print("rsu bottoms = " + str(rsu[7,200:209,-1].data))
print(" ")
print("Pressure layer: ")
print(str(pres_layer[0,200,:].data))
print("Pressure level: ")
print(str(pres_level[0,200,:].data))
print(f'half pressure layers = {0.5 * (pres_layer[0,200,1:].data + pres_layer[0,200,:-1].data)}')

print(f'levels in terms of layers = {orig_level[:]}')

if True:
    print("temp layer: ")
    print(str(temp_layer[0,200,:].data))

    print("h20 60,540,41:61 = " + str(h20[60,540,41:61].data))
    print (" ")

    print("original h20 60,540,41:61 = " + str(orig_h20[60,540,41:61]))
    print (" ")

    delta_p = orig_pres_level[:,:,1:] - orig_pres_level[:,:,:-1]

    print(f"delta_p mean = {np.mean(delta_p,axis=(0,1),dtype=np.float64)}")
    print(f"delta_p std = {np.std(delta_p,axis=(0,1))}")
    print(f"delta_p min = {np.min(delta_p,axis=(0,1))}")
    print(f"delta_p max = {np.max(delta_p,axis=(0,1))}")
    delta_delta_p = delta_p[:,:,1:] / delta_p[:,:,:-1]
    print(f"delta_delta_p mean = {np.mean(delta_delta_p,axis=(0,1))}")
    delta_co2 = orig_co2[:,:,1:] / orig_co2[:,:,:-1]
    print(f"delta_co2 mean = {np.mean(delta_co2,axis=(0,1))}")

    orig_h20 = orig_h20.reshape((-1))

    sorted_h2o = np.sort(orig_h20)
    n = sorted_h2o.shape[0]
    k = n - 1
    print(f'sorted 0 = {sorted_h2o[k]}')
    k = n - n // 1000
    print(f'sorted 0.01% = {sorted_h2o[k]}')
    k = n - n // 200
    print(f'sorted 0.5% = {sorted_h2o[k]}')
    k = n - n // 100
    print(f'sorted 1% = {sorted_h2o[k]}')
    k = n - n // 50
    print(f'sorted 2% = {sorted_h2o[k]}')
    k = n - n // 10
    print(f'sorted 10% = {sorted_h2o[k]}')
    print("")
    print("c2o 60,540,41:61 = " + str(10000.0 * c2o[60,540,41:61].data))
    print (" ")

    print("original c2o 60,540,41:61 = " + str(orig_co2[60,540,41:61] / 100.0))
    print (" ")

    print(f"original o2 = {orig_o2[:]}")
    print(f"original n2 = {orig_n2[:]}")



