from netCDF4 import Dataset
import numpy as np
import xarray as xr

data_dir       = "/home/hws/tmp/"
file_name_original   = data_dir + "CAMS_2014_RFMIPstyle.nc"
file_name_in   = data_dir + "CAMS_2009-2018_sans_2014-2015_RFMIPstyle.nc"
file_name_out  = data_dir + "RADSCHEME_data_g224_CAMS_2014.nc"
file_name_out  = data_dir + "RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.nc"
file_name_input  = data_dir + "RADSCHEME_data_g224_CAMS_2014.2.nc"
file_name_out2  = data_dir + "RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"


data_original = xr.open_dataset(file_name_original) #Dataset(file_name_in)

methane_1 = data_original.variables["methane"].data

print("!")
print(f"methane = {methane_1[1, 2, 10:20]}")

data_input = xr.open_dataset(file_name_input) #Dataset(file_name_in)

gases = data_input.variables["rrtmgp_sw_input"].data

print("!")
print(f"methane = {gases[1, 2, 10:20, 6]}")

######

h2o_1 = data_original.variables["water_vapor"].data

print("!")
print(f"h2o = {h2o_1[1, 2, 10:20]}")

print("!")
print(f"h2o = {gases[1, 2, 10:20, 2]}")

######

ozone_1 = data_original.variables["ozone"].data

print("!")
print(f"ozone = {ozone_1[1, 2, 10:20]}")

print("!")
print(f"ozone = {gases[1, 2, 10:20, 3]}")

######

v_1 = data_original.variables["carbon_dioxide"].data

print("!")
print(f"carbon_dioxide = {v_1[1, 2, 10:20]}")

print("!")
print(f"carbon_dioxide = {gases[1, 2, 10:20, 4]}")

######

v_1 = data_original.variables["nitrous_oxide"].data

print("!")
print(f"nitrous_oxide = {v_1[1, 2, 10:20]}")

print("!")
print(f"nitrous_oxide = {gases[1, 2, 10:20, 5]}")

