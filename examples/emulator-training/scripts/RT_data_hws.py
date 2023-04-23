
import sys
import numpy as np
import xarray as xr

from netCDF4 import Dataset

def absorbed_flux_to_heating_rate(absorbed_flux, delta_pressure):

    # Note cp varies with temp and pressure: https://www.ohio.edu/mechanical/thermo/property_tables/air/air_Cp_Cv.html#:~:text=The%20nominal%20values%20used%20for,v%20%3D%200.718%20kJ%2Fkg.
    cp = 1004 # J K-1  kg-1 
    g = 9.81 # m s-2
    df_dp = absorbed_flux / delta_pressure
    return -(g/cp) * 24 * 3600 * df_dp

def load_data(file_name):
    data = xr.open_dataset(file_name)

    composition = data.variables['rrtmgp_sw_input'][:].data
    (n_exp,n_col,n_layers,n_composition) = composition.shape
    n_levels = n_layers + 1
    n_samples = n_exp * n_col 

    composition = np.reshape(composition, (n_samples,n_layers,n_composition))
    t_p = composition[0:2,:,:]
    composition = composition[2:,:,:]
    n_composition = n_composition - 2

    pressure = data.variables['pres_level'][:,:,:].data 
    pressure = np.reshape(pressure,(n_samples,n_levels))

    pressure = np.reshape(pressure,(n_samples,n_levels,1))

    mass_coordinate = pressure[:,1:,:] - pressure[:,:-1,:] 

    composition = composition * mass_coordinate

    lwp = data.variables['cloud_lwp'][:].data
    iwp = data.variables['cloud_iwp'][:].data

    lwp     = np.reshape(lwp,  (n_samples,n_layers,1))    
    iwp     = np.reshape(iwp,  (n_samples,n_layers,1))

    composition = np.concatenate([composition,mass_coordinate,lwp,iwp],axis=2)
    n_composition = n_composition + 3

    mu = data.variables['mu0'][:].data 
    mu = np.reshape(mu,(n_samples,1))

    null_mu_bar = np.zeros((n_samples,0))

    surface_albedo = data.variables['sfc_alb'][:].data
    surface_albedo = surface_albedo[:,:,0]
    surface_albedo = np.reshape(surface_albedo,(n_samples,1))

    surface_absorption = np.ones(shape=(n_samples,1)) - surface_albedo

    surface = np.concatenate([surface_albedo, surface_albedo, surface_absorption, surface_absorption], axis=1)

    null_toa = np.zeros((n_samples,0))

    rsu = data.variables['rsu'][:]
    rsd = data.variables['rsd'][:]
    rsd_direct = data.variables['rsd_dir'][:]

    rsu     = np.reshape(rsu,  (n_samples,n_levels))
    rsd     = np.reshape(rsd,  (n_samples,n_levels))
    rsd_direct     = np.reshape(rsd_direct,  (n_samples,n_levels))

    toa = np.copy(rsd[:,0])

    rsu = rsu / toa
    rsd = rsd / toa
    rsd_direct = rsd_direct / toa
    absorbed_flux = rsd[:,:-1] - rsd[:,1:] + rsu[:,1:] - rsu[:,:-1]

    mass_coordinate = mass_coordinate[:,:,0]
    heating_rate = absorbed_flux_to_heating_rate (absorbed_flux, mass_coordinate)

    inputs = (t_p, composition, null_mu_bar, mu, surface, null_toa, \
    toa, mass_coordinate)

    outputs = (rsd_direct, rsd, rsu, heating_rate)

    return inputs, outputs 

def get_max():

    datadir     = "/home/hws/tmp/"
    filename_training       = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"
    filename_validation   = datadir + "/RADSCHEME_data_g224_CAMS_2014.2.nc"
    filename_testing  = datadir +  "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"

    inputs, _ = load_data(filename_validation)

    t_p, composition, null_mu_bar, mu, surface, null_toa, \
    toa, mass_coordinate = inputs

    max = np.amax (t_p, axis=-1)
    min = np.amin (t_p, axis=-1)
    print(f"t_p shape: {t_p.shape}   min = {min}    max = {max}")

if __name__ == "__main__":
    get_max()