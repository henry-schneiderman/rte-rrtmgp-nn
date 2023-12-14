from netCDF4 import Dataset
import numpy as np
import xarray as xr

data_dir       = "/data-T1/hws/tmp/"
#file_name_in   = data_dir + "CAMS_2014_RFMIPstyle.nc"

file_name_in   = data_dir + "CAMS_2009-2018_sans_2014-2015_RFMIPstyle.nc"
#file_name_out  = data_dir + "RADSCHEME_data_g224_CAMS_2014.nc"
file_name_out  = data_dir + "RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.nc"
#file_name_out2  = data_dir + "RADSCHEME_data_g224_CAMS_2014.3.nc"

file_name_out2  = data_dir + "RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.3.nc"


data_in = xr.open_dataset(file_name_in) #Dataset(file_name_in)
data_out = xr.open_dataset(file_name_out) #Dataset(file_name_out)

pres_level_in = data_in.variables["pres_level"]
oxygen_in = data_in.variables["oxygen_GM"]

#print(pres_level_in)

print("!")
print(pres_level_in[1, 2, :])



level_coord = np.arange(data_out.dims["level"])
layer_coord = np.arange(data_out.dims["layer"])
site_coord = np.arange(data_out.dims["site"])
feature_coord = np.arange(data_out.dims["feature"])
expt_coord = np.arange(data_out.dims["expt"])
gpt_coord = np.arange(224)
bnd_coord = np.arange(14)

pres_coords={"level": level_coord,
                #"layer": layer_coord,
                "site": site_coord,
                #"feature": feature_coord,
                "expt" : expt_coord}

oxygen_coords={"level": level_coord,
                #"layer": layer_coord,
                "site": site_coord,
                #"feature": feature_coord,
                "expt" : expt_coord}

data_out["pres_level"] = xr.DataArray(pres_level_in.data, coords=pres_coords, dims=data_out["rsu"].dims)
tmp_coords = {"layer" : layer_coord} #,
                #"feature" : feature_coord}
#data_out["rrtmgp_sw_input"]["layer"] = layer_coord
        
print(data_out.coords)
data_out.assign_coords(tmp_coords) 

#data_out["pres_level"].coords.to_dataset()

print(data_out.coords)

print(data_out["rsu"].shape)

data_out.to_netcdf(file_name_out2)



