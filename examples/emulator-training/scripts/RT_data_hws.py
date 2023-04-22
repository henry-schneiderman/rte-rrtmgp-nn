
import sys
import numpy as np
import xarray as xr

from netCDF4 import Dataset

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

    pressure = data.variables['pres_level'][:,:,:].data       # (nexp,ncol, nlev)
    pressure = np.reshape(pressure,(n_samples,n_levels))

    pressure = np.reshape(pressure,(n_samples,n_levels,1))
    mass_coordinate = np.zeros((n_samples,n_levels,1))
    mass_coordinate[:,:,:] = (pressure[:,1:,:] - pressure[:,:-1,:]) 

    composition = composition * mass_coordinate

    lwp = data.variables['cloud_lwp'][:].data
    iwp = data.variables['cloud_iwp'][:].data

    lwp     = np.reshape(lwp,  (n_samples,n_layers,1))    
    iwp     = np.reshape(iwp,  (n_samples,n_layers,1))

    composition = np.concatenate([composition,mass_coordinate,lwp,iwp],axis=2)
    n_composition = n_composition + 3

    mu0 = dat.variables['mu0'][:].data 