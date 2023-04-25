
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

def load_data(file_name, n_channels):
    data = xr.open_dataset(file_name)

    composition = data.variables['rrtmgp_sw_input'][:].data
    (n_exp,n_col,n_layers,n_composition) = composition.shape
    n_levels = n_layers + 1
    n_samples = n_exp * n_col 

    composition = np.reshape(composition, (n_samples,n_layers,n_composition))
    t_p = composition[:,:,0:2]

    t_p_max = np.array([320.10498, 105420.29])

    t_p_max = t_p_max.reshape((1, 1, -1))

    t_p = t_p / t_p_max

    composition = composition[:,:,2:]
    n_composition = n_composition - 2

    pressure = data.variables['pres_level'][:,:,:].data 
    pressure = np.reshape(pressure,(n_samples,n_levels))

    pressure = np.reshape(pressure,(n_samples,n_levels,1))

    delta_pressure = pressure[:,1:,:] - pressure[:,:-1,:] 

    # Treating delta_pressure as a mass coordinate
    composition = composition * delta_pressure

    lwp = data.variables['cloud_lwp'][:].data
    iwp = data.variables['cloud_iwp'][:].data

    lwp     = np.reshape(lwp,  (n_samples,n_layers,1))    
    iwp     = np.reshape(iwp,  (n_samples,n_layers,1))

    composition = np.concatenate([composition,delta_pressure,lwp,iwp],axis=2)
    n_composition = n_composition + 3

    #composition_max = np.array([7.7611343e+01, 5.3109238e-03, 1.6498107e+00, 1.3430286e-03, 7.7891685e-03, 4.0832031e+03, 2.1337291e+02, 1.9692310e+02])

    #composition_max = composition_max.reshape((1,1,-1))

    #composition = composition / composition_max

    null_lw = np.zeros((n_samples, n_layers, 0))
    null_iw = np.zeros((n_samples, n_layers, 0))

    mu = data.variables['mu0'][:].data 
    mu = np.reshape(mu,(n_samples,1))

    null_mu_bar = np.zeros((n_samples,0))

    surface_albedo = data.variables['sfc_alb'][:].data
    surface_albedo = surface_albedo[:,:,0]
    surface_albedo = np.reshape(surface_albedo,(n_samples,1))

    surface_absorption = np.ones(shape=(n_samples,1)) - surface_albedo

    surface_albedo = np.repeat(np.expand_dims(surface_albedo,axis=1),repeats=n_channels,axis=1)

    surface_absorption = np.repeat(np.expand_dims(surface_absorption,axis=1),repeats=n_channels,axis=1)

    surface = [surface_albedo, surface_albedo, surface_absorption, surface_absorption]

    null_toa = np.zeros((n_samples,0))

    flux_down_above_diffuse = np.zeros((n_samples, 2))

    rsu = data.variables['rsu'][:].data
    rsd = data.variables['rsd'][:].data
    rsd_direct = data.variables['rsd_dir'][:].data

    rsu     = rsu.reshape((n_samples,n_levels))
    rsd     = rsd.reshape((n_samples,n_levels))
    rsd_direct     = rsd_direct.reshape((n_samples,n_levels))

    toa = np.copy(rsd[:,0:1])

    rsu = rsu / toa
    rsd = rsd / toa
    rsd_direct = rsd_direct / toa
    absorbed_flux = rsd[:,:-1] - rsd[:,1:] + rsu[:,1:] - rsu[:,:-1]

    delta_pressure = np.squeeze(delta_pressure)
    heating_rate = absorbed_flux_to_heating_rate (absorbed_flux, delta_pressure)

    inputs = (t_p, composition, null_lw, null_iw, null_mu_bar, mu, surface, null_toa, \
    toa, flux_down_above_diffuse, delta_pressure)

    outputs = (rsd_direct, rsd, rsu, heating_rate)

    return inputs, outputs 

def get_max():

    datadir     = "/home/hws/tmp/"
    filename_training       = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"
    filename_validation   = datadir + "/RADSCHEME_data_g224_CAMS_2014.2.nc"
    filename_testing  = datadir +  "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"

    inputs, _ = load_data(filename_training)

    t_p, composition, null_mu_bar, mu, surface, null_toa, \
    toa, delta_pressure = inputs

    max = np.amax (t_p, axis=(0, 1))
    min = np.amin (t_p, axis=(0, 1))
    print(f"t_p shape: {t_p.shape}   min = {min}    max = {max}")

    max = np.amax (composition, axis=(0, 1))
    min = np.amin (composition, axis=(0, 1))
    print(f"composition  h2o o3 co2 n2o ch4 mass lwp iwp;")
    print(f" shape: {composition.shape}")
    print(f" min = {min}")
    print(f" max = {max}")

    #print(f'h2o ^ 0.25 = {max[0]**0.25}')
    #print(f'o3 ^ 0.25 = {max[1]**0.25}')

#if __name__ == "__main__":
#    get_max()