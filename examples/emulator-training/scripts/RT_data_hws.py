
import sys
import numpy as np
import xarray as xr
import tensorflow as tf

from netCDF4 import Dataset

g = 9.80665

def absorbed_flux_to_heating_rate(absorbed_flux, delta_pressure):

    # Note cp varies with temp and pressure: https://www.ohio.edu/mechanical/thermo/property_tables/air/air_Cp_Cv.html#:~:text=The%20nominal%20values%20used%20for,v%20%3D%200.718%20kJ%2Fkg.
    cp = 1004 # J K-1  kg-1 
    flux_div_delta_pressure = absorbed_flux / delta_pressure
    return -(g/cp) * 24 * 3600 * flux_div_delta_pressure

def load_data(file_name, n_channels):
    data = xr.open_dataset(file_name)

    composition = data.variables['rrtmgp_sw_input'][:].data
    (n_exp,n_col,n_layers,n_composition) = composition.shape
    n_levels = n_layers + 1
    n_samples = n_exp * n_col 

    composition = np.reshape(composition, (n_samples,n_layers,n_composition))
    t_p = composition[:,:,0:2]
    log_p = np.log(composition[:,:,1:2])
    t_p = np.concatenate([t_p, log_p],axis=2)

    t_p_mean = np.array([248.6, 35043.8, 8.8],dtype=np.float32)
    t_p_min = np.array([176.0, 0.0, 0.0],dtype=np.float32)
    t_p_max = np.array([320.10498, 105420.29, 11.56],dtype=np.float32)

    t_p_mean = t_p_mean.reshape((1, 1, -1))
    t_p_max = t_p_max.reshape((1, 1, -1))
    t_p_min = t_p_min.reshape((1, 1, -1))

    t_p = (t_p - t_p_mean)/ (t_p_max - t_p_min)

    composition = composition[:,:,2:]
    n_composition = n_composition - 2

    pressure = data.variables['pres_level'][:,:,:].data 
    pressure = np.reshape(pressure,(n_samples,n_levels, 1))

    delta_pressure = pressure[:,1:,:] - pressure[:,:-1,:]

    # Deriving mass coordinate from pressure difference: mass per area
    # kg / m^2
    mass_coordinate = delta_pressure / g

    composition = composition * mass_coordinate

    lwp = data.variables['cloud_lwp'][:].data / 1000.0
    iwp = data.variables['cloud_iwp'][:].data / 1000.0

    lwp     = np.reshape(lwp,  (n_samples,n_layers,1))    
    iwp     = np.reshape(iwp,  (n_samples,n_layers,1))

    composition = np.concatenate([composition,mass_coordinate,lwp,iwp],axis=2)
    n_composition = n_composition + 3

    # h2o o3 co2 n2o ch4 mass lwp iwp
    composition_max = np.array([7.9141545e+00, 5.4156350e-04, 1.6823387e-01, 1.3695081e-04, 7.9427415e-04, 4.1637085e+02, 2.1337292e-01, 1.9692309e-01],dtype=np.float32)


    #zero = np.array([0.0, 0.0, 0.0, 0.0,0.0, 0.0,100.0, 0.0,], dtype=np.float32)
    #zero = np.reshape(zero, (1, 1, -1))
    composition_max = composition_max.reshape((1,1,-1))

    composition = composition / composition_max

    null_lw = np.zeros((n_samples, n_layers, 0),dtype=np.float32)
    null_iw = np.zeros((n_samples, n_layers, 0),dtype=np.float32)

    mu = data.variables['mu0'][:].data 
    mu = np.reshape(mu,(n_samples,1,1))
    mu = np.repeat(mu,axis=1,repeats=n_layers)

    null_mu_bar = np.zeros((n_samples,0), dtype=np.float32)

    surface_albedo = data.variables['sfc_alb'][:].data
    surface_albedo = surface_albedo[:,:,0]
    surface_albedo = np.reshape(surface_albedo,(n_samples,1,1))

    surface_absorption = np.ones(shape=(n_samples,1,1), dtype=np.float32) - surface_albedo

    surface_albedo = np.repeat(np.expand_dims(surface_albedo,axis=1),repeats=n_channels,axis=1)

    surface_absorption = np.repeat(np.expand_dims(surface_absorption,axis=1),repeats=n_channels,axis=1)

    surface = [surface_albedo, surface_albedo, surface_absorption, surface_absorption]

    null_toa = np.zeros((n_samples,0), dtype=np.float32)

    flux_down_above_diffuse = np.zeros((n_samples, n_channels, 1), dtype=np.float32)

    rsu = data.variables['rsu'][:].data
    rsd = data.variables['rsd'][:].data
    rsd_direct = data.variables['rsd_dir'][:].data

    rsu     = rsu.reshape((n_samples,n_levels, 1))
    rsd     = rsd.reshape((n_samples,n_levels, 1))
    rsd_direct     = rsd_direct.reshape((n_samples,n_levels, 1))

    toa = np.copy(rsd[:,0:1,:])

    # Normalize by the toa incoming flux
    rsu = rsu / toa
    rsd = rsd / toa
    rsd_direct = rsd_direct / toa
    absorbed_flux = rsd[:,:-1,:] - rsd[:,1:,:] + rsu[:,1:,:] - rsu[:,:-1,:]

    toa = np.squeeze(toa,axis=2)


    heating_rate = absorbed_flux_to_heating_rate (absorbed_flux, delta_pressure)
    delta_pressure = np.squeeze(delta_pressure, axis=2)

    heating_rate = np.squeeze(heating_rate, axis=2)
    rsu = np.squeeze(rsu,axis=2)
    rsd = np.squeeze(rsd,axis=2)
    rsd_direct = np.squeeze(rsd_direct,axis=2)
    
    inputs = (t_p, composition, null_lw, null_iw, null_mu_bar, mu, *surface, null_toa, \
    toa, flux_down_above_diffuse, delta_pressure)

    outputs = (rsd_direct, rsd, rsu, heating_rate)

    #inputs = [x[:100] for x in inputs]
    #outputs = [x[:100] for x in outputs]

    return inputs, outputs 

def load_data_2(file_name, n_channels):
    inputs, outputs = load_data(file_name, n_channels)
    t_p, composition, null_lw, null_iw, null_mu_bar, mu, _, _, _, _, null_toa, \
    toa, flux_down_above_diffuse, delta_pressure = inputs
    i = [t_p, composition, null_lw, null_iw, null_mu_bar, mu, null_toa]
    o = [outputs[0]]
    return i, o

def load_data_full(file_name, n_channels):
    data = xr.open_dataset(file_name)
    composition = data.variables['rrtmgp_sw_input'][:].data
    (n_exp,n_col,n_layers,n_composition) = composition.shape
    n_levels = n_layers + 1
    n_samples = n_exp * n_col 

    # h2o o3 co2 n2o ch4 mass lwp iwp
    composition = np.reshape(composition, (n_samples,n_layers,n_composition))
    t_p = composition[:,:,0:2].data


    log_p = np.log(composition[:,:,1:2].data)
    t_p = np.concatenate([t_p, log_p],axis=2)

    t_p_mean = np.array([248.6, 35043.8, 8.8],dtype=np.float32)
    t_p_min = np.array([176.0, 0.0, 0.0],dtype=np.float32)
    t_p_max = np.array([320.10498, 105420.29, 11.56],dtype=np.float32)

    t_p_mean = t_p_mean.reshape((1, 1, -1))
    t_p_max = t_p_max.reshape((1, 1, -1))
    t_p_min = t_p_min.reshape((1, 1, -1))

    t_p = (t_p - t_p_mean)/ (t_p_max - t_p_min)

    h2o = composition[:,:,2:3].data / np.array([7.9141545e+02],dtype=np.float32)    # 7.9141545e+00

    h2o_sq = np.square(h2o * 10.0)

    o3 = composition[:,:,3:4].data / np.array([5.4156350],dtype=np.float32)     #  np.array([5.4156350e-04])
    co2 = composition[:,:,4:5].data / np.array([1.6823387e-01],dtype=np.float32)    # 1.6823387e-01
    n2o = composition[:,:,5:6].data / np.array([1.3695081e-05],dtype=np.float32)    # 1.3695081e-04
    ch4 = composition[:,:,6:7].data / np.array([7.9427415e-04],dtype=np.float32)    # 7.9427415e-04
    #u = composition[:,:,8:9].data / np.array([4.1637085e+02], ,dtype=np.float32)     # 4.1637085e+02

    mu = data.variables['mu0'][:].data 
    mu = np.reshape(mu,(n_samples,1,1))
    mu = np.repeat(mu,axis=1,repeats=n_layers)

    mu_bar = np.zeros((n_samples,1), dtype=np.float32)

    rsd = data.variables['rsd'][:].data
    rsd     = rsd.reshape((n_samples,n_levels, 1))
    rsd_direct = data.variables['rsd_dir'][:].data
    rsd_direct     = rsd_direct.reshape((n_samples,n_levels, 1))

    toa = np.copy(rsd[:,0:1,:])
    rsd = rsd / toa
    rsd_direct = rsd_direct / toa

    rsu = data.variables['rsu'][:].data
    rsu     = rsu.reshape((n_samples,n_levels, 1))
    rsu = rsu / toa

    absorbed_flux = rsd[:,:-1,:] - rsd[:,1:,:] + rsu[:,1:,:] - rsu[:,:-1,:]

    #lwp = data.variables['cloud_lwp'][:].data / 2.1337292e-03  #original

    lwp = data.variables['cloud_lwp'][:].data / 2.1337292e-01
    lwp     = np.reshape(lwp,  (n_samples,n_layers,1))   

    iwp = data.variables['cloud_iwp'][:].data / 1.9692309e-01
    iwp     = np.reshape(iwp,  (n_samples,n_layers,1))  

    lwp = np.concatenate([lwp, iwp], axis=2) 
    #lwp = np.expand_dims(lwp, axis=3)

    #flux_down_above_down = np.ones([n_samples,n_channels,1],dtype='float32') / n_channels

    flux_down_above_direct = np.ones([n_samples,1],dtype='float32') 
    flux_down_above_diffuse = np.zeros((n_samples, n_channels, 1), dtype=np.float32)

    rsd_direct = tf.squeeze(rsd_direct, axis=2)
    rsd = tf.squeeze(rsd, axis=2)
    rsu = tf.squeeze(rsu, axis=2)

    pressure = data.variables['pres_level'][:,:,:].data 
    pressure = np.reshape(pressure,(n_samples,n_levels))

    delta_pressure = pressure[:,1:] - pressure[:,:-1]
    delta_pressure_2 = np.reshape(np.copy(delta_pressure),(n_samples,n_layers, 1))

    # Deriving mass coordinate from pressure difference: mass per area
    # kg / m^2
    u = (delta_pressure_2 / g) / 4.1637085e+03     #4.1637085e+02

    surface_albedo = data.variables['sfc_alb'][:].data
    surface_albedo = surface_albedo[:,:,0]
    surface_albedo = np.reshape(surface_albedo,(n_samples,1,1))

    surface_absorption = np.ones(shape=(n_samples,1,1),dtype=np.float32) - surface_albedo

    surface_albedo = np.repeat(np.expand_dims(surface_albedo,axis=1),repeats=n_channels,axis=1)

    surface_absorption = np.repeat(np.expand_dims(surface_absorption,axis=1),repeats=n_channels,axis=1)

    surface = [surface_albedo, np.copy(surface_albedo), surface_absorption, np.copy(surface_absorption)]

    inputs = (mu, mu_bar, lwp, h2o, o3, co2, u, n2o, ch4, h2o_sq, t_p, *surface, flux_down_above_direct, flux_down_above_diffuse, toa[:,:,0], rsd_direct, 
              rsd, rsu, absorbed_flux, delta_pressure)
    outputs = (rsd_direct, rsd, rsu, absorbed_flux)

    return inputs, outputs

def load_data_direct(file_name, n_channels):
    tmp_inputs, tmp_outputs = load_data_full(file_name, n_channels)
    mu, mu_bar, lwp, h2o, o3, co2, u, n2o, ch4, h2o_sq, t_p,s1, s2, \
        s3, s4,flux_down_above_direct, flux_down_above_diffuse, \
            toa, rsd_direct, rsd, rsu, absorbed_flux, delta_pressure = tmp_inputs
    inputs = (mu,lwp, h2o, o3, co2, u, n2o, ch4, h2o_sq, t_p, flux_down_above_direct,  toa, rsd_direct, delta_pressure)
    outputs = (tmp_outputs[0])

    return inputs, outputs

def get_max():

    datadir     = "/home/hws/tmp/"
    filename_training       = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"
    filename_validation   = datadir + "/RADSCHEME_data_g224_CAMS_2014.2.nc"
    filename_testing  = datadir +  "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"

    inputs, _ = load_data(filename_training, n_channels=29)

    t_p, composition, null_lw, null_iw, null_mu_bar, mu, *surface, null_toa, \
    toa, flux_down_above_diffuse, delta_pressure = inputs

    max = np.amax (t_p, axis=(0, 1))
    min = np.amin (t_p, axis=(0, 1))
    mean = np.mean (t_p, axis=(0, 1))
    print(f"t_p shape: {t_p.shape}   min = {min}    max = {max}   mean = {mean}")

    max = np.amax (composition, axis=(0, 1))
    min = np.amin (composition, axis=(0, 1))
    print(f"composition  h2o o3 co2 n2o ch4 mass lwp iwp;")
    print(f" shape: {composition.shape}")
    print(f" min = {min}")
    print(f" max = {max}")

    #print(f'h2o ^ 0.25 = {max[0]**0.25}')
    #print(f'o3 ^ 0.25 = {max[1]**0.25}')

"""
if __name__ == "__main__":
    print(tf.__version__)
    get_max()  
"""
