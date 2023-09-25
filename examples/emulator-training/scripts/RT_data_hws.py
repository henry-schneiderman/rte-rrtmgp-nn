
import sys
import numpy as np
import xarray as xr


from netCDF4 import Dataset

g = 9.80665

def absorbed_flux_to_heating_rate(absorbed_flux, delta_pressure):

    # Note cp varies with temp and pressure: https://www.ohio.edu/mechanical/thermo/property_tables/air/air_Cp_Cv.html#:~:text=The%20nominal%20values%20used%20for,v%20%3D%200.718%20kJ%2Fkg.
    cp = 1004.0 # J K-1  kg-1 
    flux_div_delta_pressure = absorbed_flux / delta_pressure
    return -(g/cp) * 24.0 * 3600.0 * flux_div_delta_pressure

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

    vmr_h2o = np.copy(composition[:,:,0:1])
    m_dry = 28.964
    m_h2o =  18.01528

    mass_ratio = vmr_h2o * m_h2o / m_dry
    total_mass = mass_coordinate
    dry_mass  = total_mass / (1.0 + mass_ratio)
    h2o = dry_mass * mass_ratio

    mass_factor = np.array([47.99820, 44.0095, 44.01280, 16.0425]) / m_dry
    composition = composition[:,:,1:] * np.reshape(mass_factor, (1, 1, -1))

    if False:
        composition = composition * dry_mass
    else:
        composition = composition * dry_mass / (1.0 + composition)
        
    lwp = data.variables['cloud_lwp'][:].data 
    iwp = data.variables['cloud_iwp'][:].data 

    lwp     = np.reshape(lwp,  (n_samples,n_layers,1))    
    iwp     = np.reshape(iwp,  (n_samples,n_layers,1))

    composition = np.concatenate([h2o, composition,dry_mass,lwp,iwp],axis=2)
    n_composition = n_composition + 3

    # h2o o3 co2 n2o ch4 mass lwp iwp
    composition_max = np.array([7.9141545e+00, 5.4156350e-04, 1.6823387e-01, 1.3695081e-04, 7.9427415e-04, 4.1637085e+02, 2.1337292e-01, 1.9692309e-01],dtype=np.float32)


    #zero = np.array([0.0, 0.0, 0.0, 0.0,0.0, 0.0,100.0, 0.0,], dtype=np.float32)
    #zero = np.reshape(zero, (1, 1, -1))
    composition_max = composition_max.reshape((1,1,-1))

    #composition = composition / composition_max

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

def load_data_full(file_name, n_channels, n_coarse_code):
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

    pressure = data.variables['pres_level'][:,:,:].data 
    pressure = np.reshape(pressure,(n_samples,n_levels))

    delta_pressure = pressure[:,1:] - pressure[:,:-1]
    delta_pressure_2 = np.reshape(np.copy(delta_pressure),(n_samples,n_layers, 1))

    # Assumes vmr_h2o = moles-h2o / moles-dry-air
    vmr_h2o = np.copy(composition[:,:,2:3])
    m_dry = 28.964
    m_h2o =  18.01528

    # mass ratio = mass-h2o / mass-dry-air
    mass_ratio = vmr_h2o * m_h2o / m_dry

    # Deriving mass coordinate from pressure difference: mass per area
    # kg / m^2:
    total_mass = (delta_pressure_2 / g) 
    dry_mass  = total_mass / (1.0 + mass_ratio)
    #wet_mass = total_mass * mass_ratio / (1.0 + mass_ratio)= mass_ratio * dry_mass

    #h2o = composition[:,:,2:3] * dry_mass / np.array([7.7918200],dtype=np.float32)    
    h2o = mass_ratio * dry_mass / np.array([4.84642649],dtype=np.float32)   

    # conversion to mass ratio wrt to dry air
    # o3 co2 n2o ch4 
    mass_factor = np.array([47.99820, 44.0095, 44.01280, 16.0425]) / m_dry
    composition = composition[:,:,3:] * np.reshape(mass_factor, (1, 1, -1))

    composition = composition * dry_mass

    #composition  h2o o3 co2 n2o ch4 mass lwp iwp;
    #max = [7.7918200e+00 5.4156192e-04 1.6818701e-01 1.3688966e-04 
    # 7.9373247e-04 4.1634583e+02 2.1337291e+02 1.9692310e+02]
    #max = [7.79182005e+00 8.97450472e-04 2.55393449e-01 2.08013207e-04
    #4.39629856e-04 4.16345825e+02 2.13372910e+02 1.96923096e+02]

    h2o_sq = np.square(h2o * 10.0) #* 0.0

    o3 = composition[:,:,0:1] / np.array([8.97450472e-04],dtype=np.float32)     
    co2 = composition[:,:,1:2] / np.array([2.55393449e-01],dtype=np.float32)   
    n2o = composition[:,:,2:3] / np.array([2.08013207e-04],dtype=np.float32)  
    ch4 = composition[:,:,3:4] / np.array([4.39629856e-04],dtype=np.float32) 
    u = dry_mass / np.array([4.1634583e+02],dtype=np.float32) 

    mu = data.variables['mu0'][:].data 
    mu = np.reshape(mu,(n_samples,1,1))
    mu = np.repeat(mu,axis=1,repeats=n_layers)

    mu_bar = np.zeros((n_samples,1), dtype=np.float32)
    o2 = np.full(shape=(n_samples,n_layers, 1), fill_value=0.2, dtype=np.float32)

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

    lwp = data.variables['cloud_lwp'][:].data / 2.1337292e-00   # 2.1337292e+02
    lwp     = np.reshape(lwp,  (n_samples,n_layers,1))   

    iwp = data.variables['cloud_iwp'][:].data / 1.9692309e-00   #1.9692309e+02
    iwp     = np.reshape(iwp,  (n_samples,n_layers,1))  

    lwp = np.concatenate([lwp, iwp], axis=2) 
    #lwp = np.expand_dims(lwp, axis=3)

    #flux_down_above_down = np.ones([n_samples,n_channels,1],dtype='float32') / n_channels

    flux_down_above_direct = np.ones([n_samples,1],dtype='float32') 
    flux_down_above_diffuse = np.zeros((n_samples, n_channels, 1), dtype=np.float32)

    rsd_direct = np.squeeze(rsd_direct, axis=2)
    rsd = np.squeeze(rsd, axis=2)
    rsu = np.squeeze(rsu, axis=2)

    surface_albedo = data.variables['sfc_alb'][:].data
    surface_albedo = surface_albedo[:,:,0]
    surface_albedo = np.reshape(surface_albedo,(n_samples,1,1))

    surface_absorption = np.ones(shape=(n_samples,1,1),dtype=np.float32) - surface_albedo

    surface_albedo = np.repeat(np.expand_dims(surface_albedo,axis=1),repeats=n_channels,axis=1)

    surface_absorption = np.repeat(np.expand_dims(surface_absorption,axis=1),repeats=n_channels,axis=1)

    surface = [surface_albedo, np.copy(surface_albedo), surface_absorption, np.copy(surface_absorption)]

    coarse_code = np.ones((n_samples,n_layers,1,1), dtype=np.float32)
    #coarse_code = np.ones((n_samples,n_layers,n_channels,n_coarse_code), dtype=np.float32)
    if False:
        sigma = 0.25
        const_1 = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
        for i in range(n_channels):
            ii = i / n_channels
            for j in range(n_coarse_code):
                jj = j / n_coarse_code
                coarse_code[:,:,i,j] = const_1 * np.exp(-0.5 * np.square((ii - jj)/sigma))

    if n_coarse_code > 0:
        inputs = (mu, mu_bar, lwp, h2o, o3, co2, o2, u, n2o, ch4, h2o_sq, t_p, coarse_code, *surface, flux_down_above_direct, flux_down_above_diffuse, toa[:,:,0], rsd_direct, 
              rsd, rsu, absorbed_flux, delta_pressure)
    else:
        inputs = (mu, mu_bar, lwp, h2o, o3, co2, o2, u, n2o, ch4, h2o_sq, t_p, *surface, flux_down_above_direct, flux_down_above_diffuse, toa[:,:,0], rsd_direct, 
              rsd, rsu, absorbed_flux, delta_pressure)
    outputs = (rsd_direct, rsd, rsu, absorbed_flux)

    return inputs, outputs

def load_data_direct(file_name, n_channels):
    tmp_inputs, tmp_outputs = load_data_full(file_name, n_channels, n_coarse_code=0)
    mu, mu_bar, lwp, h2o, o3, co2, o2, u, n2o, ch4, h2o_sq, t_p, s1, s2, \
        s3, s4,flux_down_above_direct, flux_down_above_diffuse, \
            toa, rsd_direct, rsd, rsu, absorbed_flux, delta_pressure = tmp_inputs
    inputs = (mu,lwp, h2o, o3, co2, o2, u, n2o, ch4, h2o_sq * 0.0, t_p, flux_down_above_direct,  toa, rsd_direct, delta_pressure)
    outputs = (tmp_outputs[0])

    return inputs, outputs

def load_data_direct_pytorch(file_name, n_channels):
    tmp_inputs, tmp_outputs = load_data_full(file_name, n_channels, n_coarse_code=0)

    mu, mu_bar, lwp, h2o, o3, co2, o2, u, n2o, ch4, h2o_sq, t_p, s1, s2, \
        s3, s4,flux_down_above_direct, flux_down_above_diffuse, \
            toa, rsd_direct, rsd, rsu, absorbed_flux, delta_pressure = tmp_inputs
    constituents = np.concatenate([lwp,h2o,o3,co2,u,n2o,ch4],axis=2)
    x = np.concatenate([mu,t_p,constituents], axis=2)
    y = rsd_direct

    return x, y, toa, delta_pressure

def load_data_full_pytorch(file_name, n_channels):
    tmp_inputs, tmp_outputs = load_data_full(file_name, n_channels, n_coarse_code=0)

    mu, mu_bar, lwp, h2o, o3, co2, o2, u, n2o, ch4, h2o_sq, t_p, s1, s2, \
        s3, s4,flux_down_above_direct, flux_down_above_diffuse, \
            toa, rsd_direct, rsd, rsu, absorbed_flux, delta_pressure = tmp_inputs
    constituents = np.concatenate((lwp,h2o,o3,co2,u,n2o,ch4),axis=2)
    surface_properties = np.concatenate((s1,s2,s3,s4), axis=2)
    surface_properties = np.squeeze(surface_properties,axis=3)
    absorbed_flux = np.squeeze(absorbed_flux,axis=-1)
    x = np.concatenate([mu,t_p,constituents], axis=2)
    rsd_direct = np.expand_dims(rsd_direct, axis=2)
    rsd = np.expand_dims(rsd, axis=2)
    rsu = np.expand_dims(rsu, axis=2)
    y = np.concatenate((rsd_direct, rsd, rsu), axis=2)

    return  x, surface_properties, y, absorbed_flux, toa, delta_pressure    

def load_data_full_pytorch_2(file_name, n_channels):
    # Re-ordering of outputs 

    x, surface_properties, y, absorbed_flux, toa, delta_pressure = load_data_full_pytorch(file_name, n_channels)

    return x, surface_properties, toa, delta_pressure, y, absorbed_flux

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


if __name__ == "__main__":
    import tensorflow as tf
    print(tf.__version__)
    get_max()

