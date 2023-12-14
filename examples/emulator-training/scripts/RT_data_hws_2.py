import numpy as np
import random
import xarray as xr

import torch
from torch.utils.data import Dataset

def load_data_full(file, n_channels, n_coarse_code):
    data = file
    composition = data.variables['rrtmgp_sw_input'][:].data

    (n_exp,n_col,n_layers,n_composition) = composition.shape

    n_levels = n_layers + 1
    n_samples = n_exp * n_col 

    tmp_selection = data.variables['is_valid_zenith_angle'].data.astype(int)
    tmp_selection = np.reshape(tmp_selection, (n_samples))
    n_selection = np.sum(tmp_selection)
    selection = tmp_selection.astype(bool)

    composition = np.reshape(composition, (n_samples,n_layers,n_composition))
    composition = composition[selection,:,:]

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
    pressure = pressure[selection,:]

    delta_pressure = pressure[:,1:] - pressure[:,:-1]
    delta_pressure_2 = np.reshape(np.copy(delta_pressure),(n_selection,n_layers, 1))

    # Assumes vmr_h2o = moles-h2o / moles-dry-air
    m_dry = 28.964
    m_h2o =  18.01528
    g = 9.80665
    total_mass = (delta_pressure_2 / g) 

    vmr_h2o = composition[:,:,2:3]
    # mass ratio = mass-h2o / mass-dry-air
    mass_ratio = vmr_h2o * m_h2o / m_dry

    # Deriving mass coordinate from pressure difference: mass per area
    # kg / m^2:

    dry_mass  = total_mass / (1.0 + mass_ratio)
    h2o = mass_ratio * dry_mass / np.array([4.84642649],dtype=np.float32)  

    mass_factor = np.array([47.99820, 44.0095, 44.01280, 16.0425]) / m_dry
    # convert to mass ratios
    composition = composition[:,:,3:] * np.reshape(mass_factor, (1, 1, -1))
    # convert to mass
    composition = composition * dry_mass 
    h2o_sq = np.square(h2o * 10.0)

    o3 = composition[:,:,0:1] / np.array([8.97450472e-04],dtype=np.float32)     
    co2 = composition[:,:,1:2] / np.array([2.55393449e-01],dtype=np.float32)   
    n2o = composition[:,:,2:3] / np.array([2.08013207e-04],dtype=np.float32)  
    ch4 = composition[:,:,3:4] / np.array([4.39629856e-04],dtype=np.float32) 
    u = dry_mass / np.array([4.1634583e+02],dtype=np.float32) 

    mu = data.variables['mu0'][:].data 
    mu = np.reshape(mu,(n_samples,1,1))
    mu = mu[selection,:,:]
    mu = np.repeat(mu,axis=1,repeats=n_layers)

    mu_bar = np.zeros((n_selection,1), dtype=np.float32)
    o2 = np.full(shape=(n_selection,n_layers, 1), fill_value=0.2, dtype=np.float32)

    rsd = data.variables['rsd'][:].data
    rsd = rsd.reshape((n_samples,n_levels, 1))
    rsd = rsd[selection,:,:]
    rsd_direct = data.variables['rsd_dir'][:].data
    rsd_direct = rsd_direct.reshape((n_samples,n_levels, 1))
    rsd_direct = rsd_direct[selection,:,:]

    rsu = data.variables['rsu'][:].data
    rsu = rsu.reshape((n_samples,n_levels, 1))
    rsu = rsu [selection,:,:]

    # Normalize by the toa incoming flux
    toa = np.copy(rsd[:,0:1,:])
    rsd = rsd / toa
    rsd_direct = rsd_direct / toa
    rsu = rsu / toa

    absorbed_flux = rsd[:,:-1,:] - rsd[:,1:,:] + rsu[:,1:,:] - rsu[:,:-1,:]

    lwp = data.variables['cloud_lwp'][:].data / 2.1337292e-00 
    iwp = data.variables['cloud_iwp'][:].data / 1.9692309e-00

    lwp     = np.reshape(lwp,  (n_samples,n_layers,1))    
    iwp     = np.reshape(iwp,  (n_samples,n_layers,1))

    lwp = lwp[selection,:,:]
    iwp = iwp[selection,:,:]

    lwp = np.concatenate([lwp, iwp], axis=2) 

    flux_down_above_direct = np.ones([n_selection,1],dtype='float32') 
    flux_down_above_diffuse = np.zeros((n_selection, n_channels, 1), dtype=np.float32)

    rsu = np.squeeze(rsu,axis=2)
    rsd = np.squeeze(rsd,axis=2)
    rsd_direct = np.squeeze(rsd_direct,axis=2)

    surface_albedo = data.variables['sfc_alb'][:].data
    surface_albedo = surface_albedo[:,:,0]
    surface_albedo = np.reshape(surface_albedo,(n_samples,1,1))
    surface_albedo = surface_albedo[selection,:,:]

    surface_absorption = np.ones(shape=(n_selection,1,1), dtype=np.float32) - surface_albedo

    surface_albedo = np.repeat(np.expand_dims(surface_albedo,axis=1),repeats=n_channels,axis=1)

    surface_absorption = np.repeat(np.expand_dims(surface_absorption,axis=1),repeats=n_channels,axis=1)

    surface = [surface_albedo, np.copy(surface_albedo), surface_absorption, np.copy(surface_absorption)]

    coarse_code = np.ones((n_selection,n_layers,1,1), dtype=np.float32)

    if n_coarse_code > 0:
        inputs = (mu, mu_bar, lwp, h2o, o3, co2, o2, u, n2o, ch4, h2o_sq, t_p, coarse_code, *surface, flux_down_above_direct, flux_down_above_diffuse, toa[:,:,0], rsd_direct, 
              rsd, rsu, absorbed_flux, delta_pressure)
    else:
        inputs = (mu, mu_bar, lwp, h2o, o3, co2, o2, u, n2o, ch4, h2o_sq, t_p, *surface, flux_down_above_direct, flux_down_above_diffuse, toa[:,:,0], rsd_direct, 
              rsd, rsu, absorbed_flux, delta_pressure)
    outputs = (rsd_direct, rsd, rsu, absorbed_flux)

    return inputs, outputs



def load_data_full_pytorch(file, n_channels, n_coarse_code=0):
    tmp_inputs, tmp_outputs = load_data_full(file, n_channels, n_coarse_code=0)

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

def load_data_full_pytorch_2(file, n_channels):
    # Re-ordering of outputs 

    x, surface_properties, y, absorbed_flux, toa, delta_pressure = load_data_full_pytorch(file, n_channels)

    return tensorize(x), tensorize(surface_properties), tensorize(toa), tensorize(delta_pressure), tensorize(y), tensorize(absorbed_flux)
class non_sequential_access(Exception):
    "Raised when RTDataSet is not sequentially accessed"
    pass

def tensorize(np_ndarray):
    t = torch.from_numpy(np_ndarray).float()
    return t

class RTDataSet(Dataset):
    def __free_memory(self):
        del self.x 
        del self.surface_properties
        del self.toa
        del self.delta_pressure
        del self.y
        del self.absorbed_flux

    def __make_map(self):
        self.map = []
        for c in self.n_data:
            a = np.array(list(range(c)))
            random.shuffle(a)
            self.map.append(a)

    def __init__(self, input_files, n_channels, n_coarse_code=0):
        self.dt = [xr.open_dataset(f) for f in input_files]
        self.n_channels = n_channels
        self.n_coarse_code = n_coarse_code
        self.n_data_accumulated = []
        self.n_data = []
        acc = 0
        for d in self.dt:
            c = int(np.sum(d['is_valid_zenith_angle'].data))
            acc += c
            self.n_data_accumulated.append(acc)
            self.n_data.append(c)

        self.i_file = 0
        print(f"Number of valid examples = {acc}", flush=True)

    def __len__(self):
        return self.n_data_accumulated[-1]
    
    def __getitem__(self, idx):
        # Assumes data is accessed sequentially
        # Verify that it is

        if idx > self.n_data_accumulated[self.i_file]:
            raise non_sequential_access
        elif idx < 0:
            raise non_sequential_access
        else:
            if self.i_file > 0:
                if idx < self.n_data_accumulated[self.i_file - 1]:
                    print("Bad State!!!!", flush=True)
                    raise non_sequential_access

        if idx == 0:
            if hasattr(self, 'x'):
                self.__free_memory()
                del self.map
            self.__make_map()
            self.i_file = 0
            data = load_data_full_pytorch_2(self.dt[self.i_file],
                             n_channels=self.n_channels)
                             #n_coarse_code=self.n_coarse_code, 
                             #reshuffle=True)
            self.x, self.surface_properties, self.toa, self.delta_pressure, self.y, self.absorbed_flux = data
            

        elif idx == self.n_data_accumulated[self.i_file]:
            self.i_file = self.i_file + 1
            self.__free_memory()
            data = load_data_full_pytorch_2(self.dt[self.i_file],
                             n_channels=self.n_channels) 
                             #n_coarse_code=self.n_coarse_code, 
                             #reshuffle=True)
            self.x, self.surface_properties, self.toa, self.delta_pressure, self.y, self.absorbed_flux = data


        if self.i_file == 0:
            ii = self.map[self.i_file][idx]
        else:
            ii = self.map[self.i_file][idx - self.n_data_accumulated[self.i_file - 1]]

        if idx == self.n_data_accumulated[-1] - 1:
            # End of epoch
            # Note: self.i_file seems to be automatically reset to zero
            # at the end of an epoch. Can't figure out why.
            # Do it explicitly here
            self.i_file = 0

        return self.x[ii], self.surface_properties[ii], self.toa[ii], self.delta_pressure[ii], self.y[ii], self.absorbed_flux[ii]
    
if __name__ == "__main__":
    if False:
        train_input_dir = "/data-T1/hws/CAMS/processed_data/training/2008/"
        cross_input_dir = "/data-T1/hws/CAMS/processed_data/cross_validation/2008/"
        months = [str(m).zfill(2) for m in range(1,13)]
        train_input_files = [f'{train_input_dir}Flux_sw-2008-{month}.nc' for month in months]

        file = xr.open_dataset(train_input_files[0])

        tmp = load_data_full_pytorch_2(file, n_channels=30)
    else:
        a = np.array(list(range(4)))
        print(f"a = {a}")
        random.shuffle(a)
        print(f'shuffled a = {a}')