"""
Modified by Henry Schneiderman Sept. 2023.
Same as ml_loaddata.py except removed all code that didn't pertain to RNN
processing.

Python framework for developing neural network emulators of RRTMGP gas optics
scheme, the RTE radiative transfer solver, or their combination RTE+RRTMGP (a 
radiative transfer scheme).

This file provides functions for loading and preprocessing data so that it
may be used for training e.g. a neural network version of RRTMGP

@author: Peter Ukkonen
"""

import os, subprocess, argparse
import sys
import numpy as np
from netCDF4 import Dataset
# from sklearn.preprocessing import MinMaxScaler, StandardScaler

def preproc_divbymax(x,xmax=None):
    x_scaled = np.copy(x)
    if xmax is None:
        xmax = np.zeros(x.shape[-1])
        
        if np.size(x.shape)==3:
            for i in range(x.shape[-1]):
                xmax[i] =  np.max(x[:,:,i])
                x_scaled[:,:,i] =  x_scaled[:,:,i] / xmax[i]
        else:
            for i in range(x.shape[-1]):
                xmax[i] =  np.max(x[:,i])
                x_scaled[:,i] =  x_scaled[:,i] / xmax[i]
        return x_scaled, xmax
    else:
        if np.size(x.shape)==3:
            for i in range(x.shape[-1]):
                x_scaled[:,:,i] =  x_scaled[:,:,i] / xmax[i]
        else:
            for i in range(x.shape[-1]):
                x_scaled[:,i] =  x_scaled[:,i] / xmax[i]
                
        return x_scaled
    
def load_radscheme_rnn(fname, predictand='rsu_rsd', scale_p_h2o_o3=True, \
                                return_p=False, return_coldry=False):
    # Load data for training a RADIATION SCHEME (RTE+RRTMGP) emulator,
    # where inputs are vertical PROFILES of atmospheric conditions (T,p, gas concentrations)
    # and outputs (predictand) are PROFILES of broadband fluxes (upwelling and downwelling)
    # argument scale_p_h2o_o3 determines whether specific gas optics inputs
    # (pressure, H2O and O3 )  are power-scaled similarly to Ukkonen 2020 paper
    # for a more normal distribution
    
    dat = Dataset(fname)
    
    if predictand not in ['rsu_rsd']:
        sys.exit("Supported predictands (second argument) : rsu_rsd..")
            
    # temperature, pressure, and gas concentrations...
    x_gas = dat.variables['rrtmgp_sw_input'][:].data  # (nexp,ncol,nlay,ngas+2)
    (nexp,ncol,nlay,nx) = x_gas.shape
    nlev = nlay+1
    ns = nexp*ncol # number of samples (profiles)
    if scale_p_h2o_o3:
        # Log-scale pressure, power-scale H2O and O3
        x_gas[:,:,:,1] = np.log(x_gas[:,:,:,1])
        vmr_h2o = x_gas[:,:,:,2].reshape(ns,nlay)
        x_gas[:,:,:,2] = x_gas[:,:,:,2]**(1.0/4) 
        x_gas[:,:,:,3] = x_gas[:,:,:,3]**(1.0/4)
    
    # plus surface albedo, which !!!FOR THIS DATA!!! is spectrally constant
    sfc_alb = dat.variables['sfc_alb'][:].data # (nexp,ncol,ngpt)
    sfc_alb = sfc_alb[:,:,0] # (nexp,ncol)
    # plus by cosine of solar angle..
    mu0 = dat.variables['mu0'][:].data           # (nexp,ncol)
    # # ..multiplied by incoming flux
    # #  (ASSUMED CONSTANT)
    # toa_flux = dat.variables['toa_flux'][:].data # (nexp,ncol,ngpt)
    # ngpt = toa_flux.shape[-1]
    # for iexp in range(nexp):
    #     for icol in range(ncol):
    #         toa_flux[iexp,icol,:] = mu0[iexp,icol] * toa_flux[iexp,icol,:]
    
    lwp = dat.variables['cloud_lwp'][:].data
    iwp = dat.variables['cloud_iwp'][:].data

    # if predictand in ['broadband_rsu_rsd','broadband_rlu_rld']: 
    rsu = dat.variables['rsu'][:]
    rsd = dat.variables['rsd'][:]
        
    if np.size(rsu.shape) != 3:
        sys.exit("Invalid array shapes, RTE output should have 3 dimensions")
    
    # Reshape to profiles...
    x_gas   = np.reshape(x_gas,(ns,nlay,nx)) 
    lwp     = np.reshape(lwp,  (ns,nlay,1))    
    iwp     = np.reshape(iwp,  (ns,nlay,1))
    rsu     = np.reshape(rsu,  (ns,nlev))
    rsd     = np.reshape(rsd,  (ns,nlev))
    
    rsu_raw = np.copy(rsu)
    rsd_raw = np.copy(rsd)
    
    # normalize downwelling flux by the boundary condition
    rsd0    = rsd[:,0]
    rsd     = rsd / np.repeat(rsd0.reshape(-1,1), nlev, axis=1)
    # remove rsd0 from array
    rsd     = rsd[:,1:]
    # extract and remove upwelling flux at surface, this will be computed 
    # explicitly, resulting in NN outputs with consistent dimensions to input (nlay)
    rsu0    = rsu[:,-1]
    rsu     = rsu[:,0:-1]
    rsu     = rsu / np.repeat(rsd0.reshape(-1,1), nlay, axis=1)

    rsu     = rsu.reshape((ns,nlay,1))
    rsd     = rsd.reshape((ns,nlay,1))

    # Mu0 and surface albedo are also required as inputs
    # Don't know how to add constant (sequence-independent) variables,
    # so will add them as input to each sequence/level - unelegant but should work..
    mu0     = np.repeat(mu0.reshape(ns,1,1),nlay,axis=1)
    sfc_alb = np.repeat(sfc_alb.reshape(ns,1,1),nlay,axis=1)
    
    # Concatenate inputs and outputs...
    x       = np.concatenate((x_gas,lwp,iwp,mu0,sfc_alb),axis=2)
    y       = np.concatenate((rsd,rsu),axis=2)

    print( "there are {} profiles in this dataset ({} experiments, {} columns)".format(nexp*ncol,nexp,ncol))
    
    pres = dat.variables['pres_level'][:,:,:].data       # (nexp,ncol, nlev)
    pres = np.reshape(pres,(ns,nlev))
    
    if return_coldry:
        coldry = get_col_dry(vmr_h2o,pres)
    
    dat.close()
    if return_p:
        if return_coldry:
            return x,y,rsd0,rsu0,rsd_raw,rsu_raw,pres,coldry
        else:
            return x,y,rsd0,rsu0,rsd_raw,rsu_raw,pres
    
    else:
        if return_coldry:
            return x,y,rsd0,rsu0,rsd_raw,rsu_raw, coldry
        else:
            return x,y,rsd0,rsu0,rsd_raw,rsu_raw

def load_radscheme_mass_rnn(fname, predictand='rsu_rsd', scale_p_h2o_o3=True, \
                                return_p=False, return_coldry=False):
    # Same as above except loads the scaled mass of each constituent in
    # each layer
    # Load data for training a RADIATION SCHEME (RTE+RRTMGP) emulator,
    # where inputs are vertical PROFILES of atmospheric conditions (T,p, gas concentrations)
    # and outputs (predictand) are PROFILES of broadband fluxes (upwelling and downwelling)
    # argument scale_p_h2o_o3 determines whether specific gas optics inputs
    # (pressure, H2O and O3 )  are power-scaled similarly to Ukkonen 2020 paper
    # for a more normal distribution
    
    dat = Dataset(fname)
    
    if predictand not in ['rsu_rsd']:
        sys.exit("Supported predictands (second argument) : rsu_rsd..")
            
    # temperature, pressure, and gas concentrations...
    x_gas = dat.variables['rrtmgp_sw_input'][:].data  # (nexp,ncol,nlay,ngas+2)
    (nexp,ncol,nlay,nx) = x_gas.shape
    nlev = nlay+1
    ns = nexp*ncol # number of samples (profiles)

    pres = dat.variables['pres_level'][:,:,:].data       # (nexp,ncol, nlev)
    pres = np.reshape(pres,(ns,nlev))
    delta_pressure = pres[:,1:] - pres[:,:-1]
    g = 9.80665
    total_mass = delta_pressure / g
    m_dry = 28.964
    m_h2o =  18.01528

    x_gas   = np.reshape(x_gas,(ns,nlay,nx)) 
    vmr_h2o = x_gas[:,:,2]
    h2o_mass_ratio = vmr_h2o * m_h2o / m_dry
    total_mass = (delta_pressure / g) 
    dry_mass  = total_mass / (1.0 + h2o_mass_ratio)

    if scale_p_h2o_o3:
        # Log-scale pressure, power-scale H2O and O3
        x_gas[:,:,1] = np.log(x_gas[:,:,1])
        x_gas[:,:,2] = x_gas[:,:,2]*dry_mass
        x_gas[:,:,3] = x_gas[:,:,3]*dry_mass
        x_gas[:,:,4] = x_gas[:,:,4]*dry_mass
        x_gas[:,:,5] = x_gas[:,:,5]*dry_mass
        x_gas[:,:,6] = x_gas[:,:,6]*dry_mass
        x_gas[:,:,2] = x_gas[:,:,2]**(1.0/4) 
        x_gas[:,:,3] = x_gas[:,:,3]**(1.0/4)
    
    # plus surface albedo, which !!!FOR THIS DATA!!! is spectrally constant
    sfc_alb = dat.variables['sfc_alb'][:].data # (nexp,ncol,ngpt)
    sfc_alb = sfc_alb[:,:,0] # (nexp,ncol)
    # plus by cosine of solar angle..
    mu0 = dat.variables['mu0'][:].data           # (nexp,ncol)
    # # ..multiplied by incoming flux
    # #  (ASSUMED CONSTANT)
    # toa_flux = dat.variables['toa_flux'][:].data # (nexp,ncol,ngpt)
    # ngpt = toa_flux.shape[-1]
    # for iexp in range(nexp):
    #     for icol in range(ncol):
    #         toa_flux[iexp,icol,:] = mu0[iexp,icol] * toa_flux[iexp,icol,:]
    
    lwp = dat.variables['cloud_lwp'][:].data
    iwp = dat.variables['cloud_iwp'][:].data

    # if predictand in ['broadband_rsu_rsd','broadband_rlu_rld']: 
    rsu = dat.variables['rsu'][:]
    rsd = dat.variables['rsd'][:]
        
    if np.size(rsu.shape) != 3:
        sys.exit("Invalid array shapes, RTE output should have 3 dimensions")
    
    # Reshape to profiles...

    lwp     = np.reshape(lwp,  (ns,nlay,1))    
    iwp     = np.reshape(iwp,  (ns,nlay,1))
    rsu     = np.reshape(rsu,  (ns,nlev))
    rsd     = np.reshape(rsd,  (ns,nlev))
    
    rsu_raw = np.copy(rsu)
    rsd_raw = np.copy(rsd)
    
    # normalize downwelling flux by the boundary condition
    rsd0    = rsd[:,0]
    rsd     = rsd / np.repeat(rsd0.reshape(-1,1), nlev, axis=1)
    # remove rsd0 from array
    rsd     = rsd[:,1:]
    # extract and remove upwelling flux at surface, this will be computed 
    # explicitly, resulting in NN outputs with consistent dimensions to input (nlay)
    rsu0    = rsu[:,-1]
    rsu     = rsu[:,0:-1]
    rsu     = rsu / np.repeat(rsd0.reshape(-1,1), nlay, axis=1)

    rsu     = rsu.reshape((ns,nlay,1))
    rsd     = rsd.reshape((ns,nlay,1))

    # Mu0 and surface albedo are also required as inputs
    # Don't know how to add constant (sequence-independent) variables,
    # so will add them as input to each sequence/level - unelegant but should work..
    mu0     = np.repeat(mu0.reshape(ns,1,1),nlay,axis=1)
    sfc_alb = np.repeat(sfc_alb.reshape(ns,1,1),nlay,axis=1)
    
    # Concatenate inputs and outputs...
    x       = np.concatenate((x_gas,lwp,iwp,mu0,sfc_alb),axis=2)
    y       = np.concatenate((rsd,rsu),axis=2)

    print( "there are {} profiles in this dataset ({} experiments, {} columns)".format(nexp*ncol,nexp,ncol))
    

    
    if return_coldry:
        coldry = get_col_dry(vmr_h2o,pres)
    
    dat.close()
    if return_p:
        if return_coldry:
            return x,y,rsd0,rsu0,rsd_raw,rsu_raw,pres,coldry
        else:
            return x,y,rsd0,rsu0,rsd_raw,rsu_raw,pres
    
    else:
        if return_coldry:
            return x,y,rsd0,rsu0,rsd_raw,rsu_raw, coldry
        else:
            return x,y,rsd0,rsu0,rsd_raw,rsu_raw

def get_col_dry(vmr_h2o, plev):
    grav = 9.80665
    m_dry = 0.028964
    m_h2o =  0.018016
    avogad = 6.02214076e23
    delta_plev = plev[:,1:] - plev[:,0:-1]
    # Get average mass of moist air per mole of moist air
    fact = 1.0 / (1. + vmr_h2o)
    m_air = (m_dry + m_h2o * vmr_h2o) * fact
    col_dry = 10.0 * np.float64(delta_plev) * avogad * np.float64(fact) / (1000.0 * m_air * 100.0 * grav)
    return np.float32(col_dry)


def get_max():

    datadir     = "/home/hws/tmp/"
    filename_training       = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"
    filename_validation   = datadir + "/RADSCHEME_data_g224_CAMS_2014.2.nc"
    filename_testing  = datadir +  "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"

    inputs = load_radscheme_mass_rnn(filename_training)

    x,y,rsd0,rsu0,rsd_raw,rsu_raw = inputs

    max = np.amax (x, axis=(0, 1))
    min = np.amin (x, axis=(0, 1))
    print(f"composition  h2o o3 co2 n2o ch4 mass lwp iwp;")
    print(f" shape: {x.shape}")
    print(f" min = {min}")
    print(f" max = {max}")

    #print(f'h2o ^ 0.25 = {max[0]**0.25}')
    #print(f'o3 ^ 0.25 = {max[1]**0.25}')
if __name__ == "__main__":
    import tensorflow as tf
    print(tf.__version__)
    get_max()
