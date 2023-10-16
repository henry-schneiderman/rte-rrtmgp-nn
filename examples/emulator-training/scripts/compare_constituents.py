from netCDF4 import Dataset
import numpy as np
import xarray as xr

data_dir       = "/home/hws/tmp/"
#file_name_in   = data_dir + "CAMS_2014_RFMIPstyle.nc"
#file_name_in   = data_dir + "CAMS_2009-2018_sans_2014-2015_RFMIPstyle.nc"
#file_name_out  = data_dir + "RADSCHEME_data_g224_CAMS_2014.nc"
#file_name_out  = data_dir + "RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.nc"
#file_name_out2  = data_dir + "RADSCHEME_data_g224_CAMS_2014.2.nc"


file_name_out1 = "/data-T1/hws/CAMS/processed_data/CAMS_2015.final.nc"
file_name_out1a = "/data-T1/hws/CAMS/processed_data/CAMS_2015.final.n2o.nc"
file_name_out2  = data_dir + "RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"

n = 130
t = 13

data_1 = xr.open_dataset(file_name_out1) 
data_1a = xr.open_dataset(file_name_out1a) 
data_2 = xr.open_dataset(file_name_out2) 
eps = 1.0e-08
co2_1 = data_1.variables["co2"].data
o3_1 = data_1.variables["go3"].data
ch4_1 = data_1.variables["ch4"].data
q_1 = data_1.variables["q"].data
n2o_1 = data_1a.variables["N2O"].data



clwc_1 = data_1.variables["clwc"].data
ciwc_1 = data_1.variables["ciwc"].data
vmr_h2o_1 = q_1 #/ (1.0 - q_1 + eps)
#vmr_h2o_1 = q_1 / (1.0 - q_1 - ciwc_1 - clwc_1 + eps)
#no2_1 = data_1.variables["no2"].data
pressure_level_1 = data_1.variables["pressure"].data
pressure_level_1a = data_1a.variables["pressure"].data
pressure_layer_1 = 0.5 * (pressure_level_1[:,1:,:] + pressure_level_1[:,:-1,:])
pressure_layer_1a = 0.5 * (pressure_level_1a[:,1:,:] + pressure_level_1a[:,:-1,:])

pressure_layer_2 = data_2.variables["rrtmgp_sw_input"].data[:,:,:,1]
co2_2 = data_2.variables["rrtmgp_sw_input"].data[:,:,:,4]
o3_2 = data_2.variables["rrtmgp_sw_input"].data[:,:,:,3]
ch4_2 = data_2.variables["rrtmgp_sw_input"].data[:,:,:,6]
n2o_2 = data_2.variables["rrtmgp_sw_input"].data[:,:,:,5]
vmr_h2o_2 = data_2.variables["rrtmgp_sw_input"].data[:,:,:,2]
vmr_h2o_2 = np.transpose(vmr_h2o_2,(0,2,1))

m_dry = 28.964
m_h2o =  18.01528
#q_1 = q_1 * m_dry / m_h2o
vmr_h2o_2 = vmr_h2o_2 * m_h2o / m_dry

#n2o_2 = data_2.variables["rrtmgp_sw_input"].data[:,:,:,5]

eps = 1.0e-09

if True:

    #print(f"Full P0[0,:,{n}] = {co2_2[t,n,:]}\n")
    #print(f"Half P1[0,:,{n}] = {co2_1[t,:,n]}\n")

    print(f"co2 ratio should be constant = {co2_2[t,n,:] / co2_1[t,:,n] }\n")

    print(f"o3 ratio should be constant = {o3_2[t,n,:] / o3_1[t,:,n] }\n")

    print(f"ch4 ratio should be constant = {ch4_2[t,n,:] / ch4_1[t,:,n] }\n")

    print(f"vmr ratio might be constant = {vmr_h2o_2[t,:,n] / vmr_h2o_1[t,:,n] }\n")

    #print(f"min vmr ratio = {np.min(vmr_h2o_2 / vmr_h2o_1) }\n")

    #print(f"max vmr ratio  = {np.max(vmr_h2o_2 / vmr_h2o_1) }\n")

    #print(f"vmr ratios ratio = {np.min(vmr_h2o_1_old[t,:,:] / vmr_h2o_1[t,:,:]) }\n")

    print(f"Pressure layer ratio = {pressure_layer_1[t,:,n] / pressure_layer_2[t,n,:]}")

    print(f"n2o ratio = {(eps + n2o_1[t,:,n]) / (n2o_2[t,n,:] + eps)}\n")

    print(f"New n2o = {n2o_1[t,:,n]}")
    print(f"Old n2o = {n2o_2[t,n,:]}")

    print(f"New pressure ratio = {pressure_layer_1[t,:,n] / pressure_layer_1a[t,:,n]}")

    #print(f"no2 ratio should be constant = {no2_2[t,n,:] / n2o_1[t,:,n] }")