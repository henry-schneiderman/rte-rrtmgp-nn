import numpy as np
import xarray as xr

data_dir       = "/data-T1/hws/tmp/"
# https://www.climate.gov/news-features/understanding-climate/climate-change-atmospheric-carbon-dioxide

# Methane: https://climate.nasa.gov/vital-signs/methane/?intent=121
file_name_2008_01 = data_dir + '../CAMS/processed_data/training/2008/Flux_sw-2008-01.nc'
dt_2008 = xr.open_dataset(file_name_2008_01)

composition_2008 = dt_2008['rrtmgp_sw_input'].data

(n_exp,n_col,n_layers, n_constituents) = composition_2008.shape


n_levels = n_layers + 1
n_samples = n_exp * n_col 

composition_2008 = np.reshape(composition_2008, (n_samples, n_layers, n_constituents))

pressure = dt_2008['pres_level'][:,:,:].data 
pressure = np.reshape(pressure,(n_samples,n_levels))

delta_pressure = pressure[:,1:] - pressure[:,:-1]
delta_pressure = np.reshape(delta_pressure,(n_samples,n_layers, 1))

# Assumes vmr_h2o = moles-h2o / moles-dry-air
m_dry = 28.964
m_h2o =  18.01528
g = 9.80665
total_mass = (delta_pressure / g) 

vmr_h2o = composition_2008[:,:,2:3]
# mass ratio = mass-h2o / mass-dry-air
mass_ratio = vmr_h2o * m_h2o / m_dry

# Deriving mass coordinate from pressure difference: mass per area
# kg / m^2:

dry_mass  = total_mass / (1.0 + mass_ratio)
h2o = mass_ratio * dry_mass / np.array([4.84642649],dtype=np.float32)  

mass_factor = np.array([47.99820, 44.0095, 44.01280, 16.0425]) / m_dry
# convert to mass ratios
composition_2008 = composition_2008[:,:,3:] * np.reshape(mass_factor, (1, 1, -1))
# convert to mass
composition_2008 = composition_2008 * dry_mass 

total_co_2008 = np.sum(composition_2008[:,:,1]) / float (n_samples)
print(f"sum of co2_2008 = {total_co_2008}")

total_o3_2008 = np.sum(composition_2008[:,:,0])/ float (n_samples)
print(f"sum of o3_2008 = {total_o3_2008}")

total_ch4_2008 = np.sum(composition_2008[:,:,3])/ float (n_samples)
print(f"sum of ch4_2008 = {total_ch4_2008}")

########

if True:


    file_name_2020_01 = data_dir + '../CAMS/processed_data/testing/2020/Flux_sw-2020-01.nc'
    dt_2020 = xr.open_dataset(file_name_2020_01)

    composition_2020 = dt_2020['rrtmgp_sw_input'].data

    (n_exp,n_col,n_layers, n_constituents) = composition_2020.shape

    n_levels = n_layers + 1
    n_samples = n_exp * n_col 

    composition_2020 = np.reshape(composition_2020, (n_samples, n_layers, n_constituents))

    pressure = dt_2020['pres_level'][:,:,:].data 
    pressure = np.reshape(pressure,(n_samples,n_levels))

    delta_pressure = pressure[:,1:] - pressure[:,:-1]
    delta_pressure = np.reshape(delta_pressure,(n_samples,n_layers, 1))

    # Assumes vmr_h2o = moles-h2o / moles-dry-air
    m_dry = 28.964
    m_h2o =  18.01528
    g = 9.80665
    total_mass = (delta_pressure / g) 

    vmr_h2o = composition_2020[:,:,2:3]
    # mass ratio = mass-h2o / mass-dry-air
    mass_ratio = vmr_h2o * m_h2o / m_dry

    # Deriving mass coordinate from pressure difference: mass per area
    # kg / m^2:

    dry_mass  = total_mass / (1.0 + mass_ratio)
    h2o = mass_ratio * dry_mass / np.array([4.84642649],dtype=np.float32)  

    mass_factor = np.array([47.99820, 44.0095, 44.01280, 16.0425]) / m_dry
    # convert to mass ratios
    composition_2020 = composition_2020[:,:,3:] * np.reshape(mass_factor, (1, 1, -1))
    # convert to mass
    composition_2020 = composition_2020 * dry_mass 

    total_co_2020 = np.sum(composition_2020[:,:,1])/ float (n_samples)
    print(f"sum of co2_2020 = {total_co_2020}")

    total_o3_2020 = np.sum(composition_2020[:,:,0])/ float (n_samples)
    print(f"sum of o3_2020 = {total_o3_2020}")

    total_ch4_2020 = np.sum(composition_2020[:,:,3])/ float (n_samples)
    print(f"sum of ch4_2020 = {total_ch4_2020}")

    print(f"co2 ratio of 2020 / 2008 = {total_co_2020 / total_co_2008}")
    print(f"o3 ratio of 2020 / 2008 = {total_o3_2020 / total_o3_2008}")
    print(f"ch4 ratio of 2020 / 2008 = {total_ch4_2020 / total_ch4_2008}")


if True:

    file_name_2015_01 = data_dir + '../CAMS/processed_data/testing/2015/Flux_sw-2015-01.nc'
    dt_2015 = xr.open_dataset(file_name_2015_01)

    composition_2015 = dt_2015['rrtmgp_sw_input'].data

    (n_exp,n_col,n_layers, n_constituents) = composition_2015.shape

    n_levels = n_layers + 1
    n_samples = n_exp * n_col 

    composition_2015 = np.reshape(composition_2015, (n_samples, n_layers, n_constituents))

    pressure = dt_2015['pres_level'][:,:,:].data 
    pressure = np.reshape(pressure,(n_samples,n_levels))

    delta_pressure = pressure[:,1:] - pressure[:,:-1]
    delta_pressure = np.reshape(delta_pressure,(n_samples,n_layers, 1))

    # Assumes vmr_h2o = moles-h2o / moles-dry-air
    m_dry = 28.964
    m_h2o =  18.01528
    g = 9.80665
    total_mass = (delta_pressure / g) 

    vmr_h2o = composition_2015[:,:,2:3]
    # mass ratio = mass-h2o / mass-dry-air
    mass_ratio = vmr_h2o * m_h2o / m_dry

    # Deriving mass coordinate from pressure difference: mass per area
    # kg / m^2:

    dry_mass  = total_mass / (1.0 + mass_ratio)
    h2o = mass_ratio * dry_mass / np.array([4.84642649],dtype=np.float32)  

    mass_factor = np.array([47.99820, 44.0095, 44.01280, 16.0425]) / m_dry
    # convert to mass ratios
    composition_2015 = composition_2015[:,:,3:] * np.reshape(mass_factor, (1, 1, -1))
    # convert to mass
    composition_2015 = composition_2015 * dry_mass 

    total_co_2015 = np.sum(composition_2015[:,:,1])/ float (n_samples)
    print(f"sum of co2_2015 = {total_co_2015}")

    total_o3_2015 = np.sum(composition_2015[:,:,0])/ float (n_samples)
    print(f"sum of o3_2015 = {total_o3_2015}")

    total_ch4_2015 = np.sum(composition_2015[:,:,3])/ float (n_samples)
    print(f"sum of ch4_2015 = {total_ch4_2015}")

    print(f"co2 ratio of 2015 / 2008 = {total_co_2015 / total_co_2008}")
    print(f"o3 ratio of 2015 / 2008 = {total_o3_2015 / total_o3_2008}")
    print(f"ch4 ratio of 2015 / 2008 = {total_ch4_2015 / total_ch4_2008}")

if True:

    file_name_2009_01 = data_dir + '../CAMS/processed_data/testing/2009/Flux_sw-2009-01.nc'
    dt_2009 = xr.open_dataset(file_name_2009_01)

    composition_2009 = dt_2009['rrtmgp_sw_input'].data

    (n_exp,n_col,n_layers, n_constituents) = composition_2009.shape

    n_levels = n_layers + 1
    n_samples = n_exp * n_col 

    composition_2009 = np.reshape(composition_2009, (n_samples, n_layers, n_constituents))

    pressure = dt_2009['pres_level'][:,:,:].data 
    pressure = np.reshape(pressure,(n_samples,n_levels))

    delta_pressure = pressure[:,1:] - pressure[:,:-1]
    delta_pressure = np.reshape(delta_pressure,(n_samples,n_layers, 1))

    # Assumes vmr_h2o = moles-h2o / moles-dry-air
    m_dry = 28.964
    m_h2o =  18.01528
    g = 9.80665
    total_mass = (delta_pressure / g) 

    vmr_h2o = composition_2009[:,:,2:3]
    # mass ratio = mass-h2o / mass-dry-air
    mass_ratio = vmr_h2o * m_h2o / m_dry

    # Deriving mass coordinate from pressure difference: mass per area
    # kg / m^2:

    dry_mass  = total_mass / (1.0 + mass_ratio)
    h2o = mass_ratio * dry_mass / np.array([4.84642649],dtype=np.float32)  

    mass_factor = np.array([47.99820, 44.0095, 44.01280, 16.0425]) / m_dry
    # convert to mass ratios
    composition_2009 = composition_2009[:,:,3:] * np.reshape(mass_factor, (1, 1, -1))
    # convert to mass
    composition_2009 = composition_2009 * dry_mass 

    total_co_2009 = np.sum(composition_2009[:,:,1])/ float (n_samples)
    print(f"sum of co2_2009 = {total_co_2009}")

    total_o3_2009 = np.sum(composition_2009[:,:,0])/ float (n_samples)
    print(f"sum of o3_2009 = {total_o3_2009}")

    total_ch4_2009 = np.sum(composition_2009[:,:,3])/ float (n_samples)
    print(f"sum of ch4_2009 = {total_ch4_2009}")

    print(f"co2 ratio of 2009 / 2008 = {total_co_2009 / total_co_2008}")
    print(f"o3 ratio of 2009 / 2008 = {total_o3_2009 / total_o3_2008}")
    print(f"ch4 ratio of 2009 / 2008 = {total_ch4_2009 / total_ch4_2008}")
