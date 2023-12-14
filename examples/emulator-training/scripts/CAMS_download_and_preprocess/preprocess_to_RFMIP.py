#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Need to fix cloud fraction for long-wave radiation

import os
import xarray as xr
import numpy as np

from regrid_gases import regrid_gases
from regrid_n2o import regrid_n2o

rd = 2.8705e2 # gas constant of dry air [J/kg/K]
rv = 4.6150e2 # gas constant water [J/kg/K]
lv = 2.5e6 # Latent haet of vaporization [J / Kg]
ld = 2.83e6 # Latent heat of deposition [J / Kg]
e0 = 611.3 #[Pa]
t0 = 273.15 # [K]
m_dry = 28.9644 # more signficant digits
m_h2o =  18.01528
m_o3 = 47.9982
m_ch4 = 16.0425
m_co2 = 44.0095
m_co = 28.0101
m_no2 = 46.0055
co2_scale_factor = 1.0e-6
no2_scale_factor = 1.0e-6
n2o_scale_factor = 1.0e-9
ch4_scale_factor = 1.0e-9
mole_fraction_o2 = 0.029
mole_fraction_n2 = 0.781

def gfs_cloud_fraction(RH, saturation_specific_humidity,cloud_condensate):
    # https://dtcenter.ucar.edu/gmtb/users/ccpp/docs/sci_doc/group__module__radiation__clouds.html
    print(f"min qs = {np.min(saturation_specific_humidity)}")
    print(f"min rh = {np.min(RH)}")
    print(f"max rh = {np.max(RH)}")
    ql = -cloud_condensate * 100.0
    denom = ((1.0 - RH) * saturation_specific_humidity) ** 0.49
    sigma = RH ** 0.25 * (1.0 - np.exp(ql / denom))
    return sigma

def saturation_vapor_pressure(t):
    # Clausius-Clapeyron equation for water only
    # https://geo.libretexts.org/Bookshelves/Meteorology_and_Climate_Science/Practical_Meteorology_(Stull)/04%3A_Water_Vapor/4.00%3A_Vapor_Pressure_at_Saturation
    return e0 * np.exp((lv / rv) * ((1.0/t0) - (1.0/t)))
    #return e0 * np.exp((ld / rv) * ((1.0/t0) - (1.0/t)))    

def saturation_vapor_pressure_2(T):
    """
    Created on Mon Sep 13 15:01:22 2021

    @author: peter
    """
    ''' get sateration pressure (units [Pa]) for a given air temperature (units [K])'''
    TK = 273.15
    e1 = 101325.0
    logTTK = np.log10(T/TK)
    esat =  e1*10**(10.79586*(1-TK/T)-5.02808*logTTK+ 1.50474*1e-4*(1.-10**(-8.29692*(T/TK-1)))+ 0.42873*1e-3*(10**(4.76955*(1-TK/T))-1)-2.2195983) 
    return esat

def relative_humidity(w, es, p):
    # ws = es * m_h2o / (m_dry(p - es))
    # w = e * m_h2o / (m_dry(p - e))
    # https://earthscience.stackexchange.com/questions/5076/how-to-calculate-specific-humidity-with-relative-humidity-temperature-and-pres
    # https://vortex.plymouth.edu/~stmiller/stmiller_content/Publications/AtmosRH_Equations_Rev.pdf

    e = w * p / ((m_h2o / m_dry) + w) #correct
    #e = w * rv * p / (rd + w * rv)
    rh = e / es
    return rh

def saturation_specific_humidity(rh,w):
    # specific humidity: q = mv / (mv + md) 
    # mass mixing ratio: w = mv / md
    # q = w / (w + 1)  [dividing right side by md]
    # w = q / (1 - q)
    # rh = w / ws
    ws = w / rh
    return ws / (ws + 1.0)

def add_coords_to_RFMIP(CAMS_file_name, output_file_name):
    dt_input   = xr.open_dataset(CAMS_file_name)

    dt_input = dt_input.rename({"cell":"site"})

    sites = dt_input.dims['site']
    layers = dt_input.dims['layer']

    dt_input = dt_input.assign_coords({"site":range(1,sites+1),"layer":range(1,layers+1), "level":range(1,layers+2)})

    dt_input.to_netcdf(output_file_name)
    dt_input.close()

def preprocess_to_RFMIP(CAMS_file_name, output_file_name):

    dt_input   = xr.open_dataset(CAMS_file_name)

    dt_input = dt_input.rename({"skt": "surface_temperature", "N2O":'nitrous_oxide',
                                "t":"temp_layer", "fal":"surface_albedo","go3":"ozone", "ch4":"methane", "co2":"carbon_dioxide",
                                "co":"carbon_monoxide",
                                "no2":"nitrogen_dioxide"}) 

    times = dt_input.coords['time']
    sites = dt_input.coords['site']
    layers = dt_input.coords['layer']
    levels = dt_input.coords['level']

    dt_input['pres_layer'] = dt_input['pres_layer'].transpose("time", "site", "layer")
    pres_layer = dt_input['pres_layer'].data
    dt_input['pres_level'] = dt_input['pres_level'].transpose("time", "site", "level")
    pres_level = dt_input['pres_level'].data
    dt_input['temp_layer'] = dt_input['temp_layer'].transpose("time", "site", "layer")
    temp_layer = dt_input['temp_layer'].data
    temp_surface = dt_input['surface_temperature'].data
    temp_level = np.zeros((len(times.data),len(sites.data),len(levels.data)))
    temp_level[:,:,-1] = 0.5 * (temp_surface[:,:] + temp_layer[:,:,-1])
    temp_level[:,:,1:-1] = 0.5 * (temp_layer[:,:,0:-1] + temp_layer[:,:,1:])
    temp_level[:,:,0] = temp_layer[:,:,0] + (pres_level[:,:,0] - pres_layer[:,:,0]) * \
    (temp_layer[:,:,1]-temp_layer[:,:,0]) / (pres_layer[:,:,1]-pres_layer[:,:,0])
    temp_level = xr.DataArray(temp_level, dims=("time","site","level"), name="temp_level")
    temp_level.attrs["units"] = "K"
    temp_level.attrs["standard_name"] = "air_temperature"
    dt_input["temp_level"] = temp_level

    dt_input['ciwc'] = dt_input['ciwc'].transpose("time", "site", "layer")
    ciwc = dt_input['ciwc'].data
    dt_input['clwc'] = dt_input['clwc'].transpose("time", "site", "layer")
    clwc = dt_input['clwc'].data

    es = saturation_vapor_pressure_2(temp_layer)

    # specific humidity
    q = dt_input['q'].data

    # swap layer and site axes
    q = np.transpose(q,(0,2,1))
    print(f"min q = {np.min(q)}")
    q[q<0] = 0.0

    # mass mixing ratio: water-vapor-mass / dry-air-mass
    # Also known as "humidity mixing ratio (mixr)""
    w = q / (1 - q)

    rh = relative_humidity(w, es, pres_layer)
    qs = saturation_specific_humidity(rh, w)
    #cloud_fraction = gfs_cloud_fraction(rh, qs, clwc + ciwc)
    cloud_fraction = np.zeros((len(times.data),len(sites.data),len(layers.data)))

    cloud_fraction = xr.DataArray(cloud_fraction, dims=("time","site","layer"), name="cloud_fraction")
    cloud_fraction.attrs["units"] = "1"
    cloud_fraction.attrs["long_name"] = "Cloud Fraction"
    cloud_fraction.attrs["comment"] = "Computed from https://dtcenter.ucar.edu/gmtb/users/ccpp/docs/sci_doc/group__module__radiation__clouds.html"
    dt_input['cloud_fraction'] = cloud_fraction

    surface_emissivity = np.full(len(sites.data), fill_value=0.5)
    surface_emissivity = xr.DataArray(surface_emissivity, dims=("site"), name="surface_emissivity")
    surface_emissivity.attrs["units"] = "1"
    surface_emissivity.attrs["standard_name"] = "surface_longwave_emissivity"
    dt_input['surface_emissivity'] = surface_emissivity

    # mole mixing ratio
    vmr_h2o = w * m_dry / m_h2o

    #vmr_h2o = xr.DataArray(vmr_h2o, coords={"time": times.data, "site":sites.data, "layer":layers.data}, name="water_vapor")
    vmr_h2o = xr.DataArray(vmr_h2o, dims=("time", "site", "layer"), name="water_vapor")
    vmr_h2o.attrs["units"] = "1"
    vmr_h2o.attrs["CDI_grid_type"] = "unstructured"
    #vmr_h2o.attrs["standard_name"] = "mole_fraction_of_water_vapor_in_air"
    vmr_h2o.attrs["name"] = "vmr of water vapor"
    vmr_h2o.attrs["comment"] = "moles water vapor / moles of dry air"


    dt_input["water_vapor"] = vmr_h2o
    dt_input = dt_input.drop_vars("q") #does not work!


    # Ozone mass mixing ratio
    o3 = dt_input['ozone'].transpose("time", "site", "layer")
    vmr_o3 = o3.data * m_dry / m_o3
    o3.data = vmr_o3
    o3.attrs["name"] = "vmr of ozone (full chemistry scheme)"
    o3.attrs["comment"] = "moles ozone / moles of dry air"
    o3.attrs["units"] = "1"
    del o3.attrs["long_name"]
    del o3.attrs["standard_name"]
    del o3.attrs["param"]
    #del o3.attrs["coordinates"]
    # Need to delete "param"
    dt_input["ozone"] = o3

    dt_input['methane'] = dt_input['methane'].transpose("time", "site", "layer")
    methane = dt_input['methane']
    vmr_methane = methane.data * (m_dry / m_ch4) / ch4_scale_factor
    methane.data = vmr_methane
    methane.attrs["name"] = "vmr of methane"
    methane.attrs["comment"] = "moles methane / moles of dry air"
    methane.attrs["units"] = f"{ch4_scale_factor}"
    del methane.attrs["long_name"]
    del methane.attrs["standard_name"]
    del methane.attrs["param"]
    #del methane.attrs["coordinates"]
    dt_input["methane"] = methane 

    dt_input['carbon_dioxide'] = dt_input['carbon_dioxide'].transpose("time", "site", "layer")
    carbon_dioxide = dt_input['carbon_dioxide']
    vmr_carbon_dioxide = carbon_dioxide.data * (m_dry / m_co2) / co2_scale_factor
    carbon_dioxide.data = vmr_carbon_dioxide
    carbon_dioxide.attrs["name"] = "vmr of carbon_dioxide"
    carbon_dioxide.attrs["comment"] = "moles carbon_dioxide / moles of dry air"
    carbon_dioxide.attrs["units"] = f'{co2_scale_factor}'
    del carbon_dioxide.attrs["standard_name"]
    del carbon_dioxide.attrs["param"]
    #del carbon_dioxide.attrs["coordinates"]
    dt_input["carbon_dioxide"] = carbon_dioxide

    dt_input['carbon_monoxide'] = dt_input['carbon_monoxide'].transpose("time", "site", "layer")
    carbon_monoxide = dt_input['carbon_monoxide']
    vmr_carbon_monoxide = carbon_monoxide.data * m_dry / m_co
    carbon_monoxide.data = vmr_carbon_monoxide
    carbon_monoxide.attrs["name"] = "vmr of carbon_monoxide"
    carbon_monoxide.attrs["comment"] = "moles carbon_monoxide / moles of dry air"
    carbon_monoxide.attrs["units"] = "1"
    del carbon_monoxide.attrs["standard_name"]
    del carbon_monoxide.attrs["long_name"]
    del carbon_monoxide.attrs["param"]
    #del carbon_monoxide.attrs["coordinates"]
    dt_input["carbon_monoxide"] = carbon_monoxide

    dt_input['nitrogen_dioxide'] = dt_input['nitrogen_dioxide'].transpose("time", "site", "layer")
    nitrogen_dioxide = dt_input['nitrogen_dioxide']
    vmr_nitrogen_dioxide = nitrogen_dioxide.data * (m_dry / m_co2) / no2_scale_factor 
    nitrogen_dioxide.data = vmr_nitrogen_dioxide
    nitrogen_dioxide.attrs["name"] = "vmr of nitrogen_dioxide"
    nitrogen_dioxide.attrs["comment"] = "moles nitrogen_dioxide / moles of dry air"
    nitrogen_dioxide.attrs["units"] = f"{no2_scale_factor}"
    del nitrogen_dioxide.attrs["standard_name"]
    del nitrogen_dioxide.attrs["long_name"]
    del nitrogen_dioxide.attrs["param"]
    #del nitrogen_dioxide.attrs["coordinates"]
    dt_input["nitrogen_dioxide"] = nitrogen_dioxide

    dt_input['nitrous_oxide'] = dt_input['nitrous_oxide'].transpose("time", "site", "layer")
    nitrous_oxide = dt_input['nitrous_oxide']
    # Nitrous Oxide is the form of vmr and has already been scaled
    nitrous_oxide.attrs["name"] = "vmr of nitrous_oxide"
    nitrous_oxide.attrs["comment"] = "moles nitrous_oxide / moles of dry air"
    nitrous_oxide.attrs["units"] = f"{n2o_scale_factor}"

    #del nitrous_oxide.attrs["coordinates"]
    dt_input["nitrous_oxide"] = nitrous_oxide

    oxygen_GM = np.full((len(times.data)),fill_value=mole_fraction_o2)
    oxygen_GM = xr.DataArray(oxygen_GM, dims=("time"), name="oxygen_GM")
    oxygen_GM.attrs["units"] = "1"
    oxygen_GM.attrs["standard_name"] = "mole_fraction_of_molecular_oxygen_in_air"
    dt_input["oxygen_GM"] = oxygen_GM

    nitrogen_GM = np.full((len(times.data)),fill_value=mole_fraction_o2)
    nitrogen_GM = xr.DataArray(nitrogen_GM, dims=("time"), name="nitrogen_GM")
    nitrogen_GM.attrs["units"] = "1"
    nitrogen_GM.attrs["standard_name"] = "mole_fraction_of_molecular_nitrogen_in_air"
    dt_input["nitrogen_GM"] = nitrogen_GM

    dt_input.to_netcdf(output_file_name)
    dt_input.close()

def fix_methane(CAMS_file_name, output_file_name):

    dt_input   = xr.open_dataset(CAMS_file_name)

    methane = dt_input['methane']
    vmr_methane = methane.data / ch4_scale_factor
    methane.data = vmr_methane
    methane.attrs["units"] = f"{ch4_scale_factor}"
    dt_input["methane"] = methane 

    dt_input.to_netcdf(output_file_name)
    dt_input.close()

def add_is_valid_zenith_angle(input_file_name, output_file_name):
    dt = xr.open_dataset(input_file_name)
    mu0 = dt["mu0"].data
    shape = mu0.shape
    is_valid_zenith_angle = xr.DataArray(np.ones((shape)), dims=("expt","time"), name="is_valid_zenith_angle")
    is_valid_zenith_angle.attrs["long_name"] = "True if zenith angle is less than 90 degrees"
    dt["is_valid_zenith_angle"] = is_valid_zenith_angle
    data = dt["rrtmgp_sw_input"].data
    data[:,:,:,-1] = data[:,:,:,-1] * 1.0e-9
    dt['rrtmgp_sw_input'].data = data
    dt.to_netcdf(output_file_name)
    dt.close()


def cleanup_intermediate_files(dir, year, month):

    # Do NOT remove {dir}/CAMS_{year}-{month}.final.nc
    # or {dir}/CAMS_{year}-{month}.4.nc

    cmd = f'rm -f {dir}/CAMS_{year}-{month}.3.nc'
    os.system(cmd)

    cmd = f'rm -f {dir}/CAMS_{year}-{month}.2.nc {dir}/CAMS_{year}-{month}.1.nc'
    os.system(cmd)

    cmd = f'rm -f {dir}/tmp.n2o.*.nc'
    os.system(cmd)

    cmd = f'rm -f {dir}/CAMS_eac*.nc'
    os.system(cmd)

    cmd = f'rm -f {dir}/CAMS_egg*.nc'
    os.system(cmd)

    cmd = f'rm -f {dir}/CAMS_eac*.grb'
    os.system(cmd)

    cmd = f'rm -f {dir}/CAMS_egg*.grb'
    os.system(cmd)

    cmd = f'rm -f {dir}/CAMS*pressure*.nc'
    os.system(cmd)

    cmd = f'rm -f {dir}/era5*.nc'
    os.system(cmd)

    cmd = f'rm -f {dir}/era5*.grb'
    os.system(cmd)

    cmd = f'rm -f {dir}/cams73*.nc'
    os.system(cmd)


if __name__ == "__main__":

    year = '2015'
    mode = 'testing'
    original_data_dir = '/data-T1/hws/CAMS/original_data/'
    processed_data_dir = '/data-T1/hws/CAMS/processed_data/'
    months = [str(m).zfill(2) for m in range(1,13)]

    for month in months:

        dir = f'{processed_data_dir}{mode}/{year}/{month}/'

        if False:

            CAMS_file_name = dir + f'CAMS_{year}-{month}.3.nc'
            CAMS_file_name_2 = dir + f'CAMS_{year}-{month}.4.nc'
            output_file_name = dir + f'CAMS_{year}-{month}.final.nc'
            
            regrid_gases('/data-T1/hws/CAMS/',mode, month, year, use_st=False)
            regrid_n2o(original_data_dir, processed_data_dir, mode, month, year)
            add_coords_to_RFMIP(CAMS_file_name, CAMS_file_name_2)
            preprocess_to_RFMIP(CAMS_file_name_2, output_file_name)


        elif True:
            year = '2015'
            mode = 'testing'
            dir = f'{processed_data_dir}{mode}/{year}/{month}/'
            cleanup_intermediate_files(dir, year, month)
        else:
            add_is_valid_zenith_angle("/data-T1/hws/tmp/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc", 
                                    "/data-T1/hws/tmp/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.2.nc")

