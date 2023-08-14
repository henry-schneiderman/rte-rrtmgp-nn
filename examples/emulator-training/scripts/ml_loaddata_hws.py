#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted from ml_loaddata.py by Henry Schneiderman

Python framework for developing neural network emulators of RRTMGP gas optics
scheme, the RTE radiative transfer solver, or their combination RTE+RRTMGP (a 
radiative transfer scheme).

This file provides functions for loading and preprocessing data so that it
may be used for training e.g. a neural network version of RRTMGP

@author: Peter Ukkonen
"""


import sys
import numpy as np

from netCDF4 import Dataset
# from sklearn.preprocessing import MinMaxScaler, StandardScaler

# input scaling coefficients for RRTMGP-NN - these should probably be put in an
# external file 
ymeans_sw_abs = np.array([3.64390580e-04, 4.35663940e-04, 4.98018635e-04, 5.77545608e-04,
       6.80800469e-04, 7.98740832e-04, 9.35279648e-04, 1.16656872e-03,
       1.58452173e-03, 1.86584354e-03, 1.99465151e-03, 2.16701976e-03,
       2.41802959e-03, 2.82146805e-03, 3.48183908e-03, 4.09035478e-03,
       3.24113556e-04, 3.74707161e-04, 4.17389121e-04, 4.57456830e-04,
       4.98836453e-04, 5.49621007e-04, 6.13025972e-04, 7.00094330e-04,
       8.49446864e-04, 9.81244841e-04, 1.03521883e-03, 1.10830076e-03,
       1.21134310e-03, 1.31195760e-03, 1.39195414e-03, 1.45876186e-03,
       4.28469037e-04, 5.20646572e-04, 6.22550666e-04, 7.22033263e-04,
       8.23189737e-04, 9.40993312e-04, 1.06700219e-03, 1.21110224e-03,
       1.45994173e-03, 1.69180706e-03, 1.79443962e-03, 1.92319078e-03,
       2.08631344e-03, 2.33873748e-03, 2.59446981e-03, 2.72375043e-03,
       2.89453164e-04, 3.26674141e-04, 3.59543861e-04, 3.93101625e-04,
       4.30800777e-04, 4.71213047e-04, 5.19042369e-04, 5.83244429e-04,
       6.85371691e-04, 7.79234222e-04, 8.19451292e-04, 8.65648268e-04,
       9.23064887e-04, 1.00047945e-03, 1.08587136e-03, 1.09644048e-03,
       2.97961291e-04, 3.59470578e-04, 4.12600290e-04, 4.63586446e-04,
       5.14341518e-04, 5.67600771e-04, 6.28228823e-04, 7.09333457e-04,
       8.51527497e-04, 9.86502739e-04, 1.04738004e-03, 1.12565351e-03,
       1.23939372e-03, 1.37201988e-03, 1.52829266e-03, 1.63304247e-03,
       2.70115474e-04, 3.13259574e-04, 3.55943455e-04, 4.24126105e-04,
       5.12095110e-04, 5.70286124e-04, 6.33014424e-04, 7.20241107e-04,
       8.71218799e-04, 1.01254229e-03, 1.07443938e-03, 1.14905764e-03,
       1.24553905e-03, 1.37227518e-03, 1.50288385e-03, 1.59471075e-03,
       2.49097910e-04, 3.04214394e-04, 3.61522223e-04, 4.22842306e-04,
       4.84134798e-04, 5.45158167e-04, 6.10073039e-04, 6.93159061e-04,
       8.35262705e-04, 9.70904832e-04, 1.03256141e-03, 1.10617711e-03,
       1.19320781e-03, 1.30544091e-03, 1.43013091e-03, 1.53011072e-03,
       3.03399313e-04, 3.29578412e-04, 3.45331355e-04, 3.61360639e-04,
       3.78781464e-04, 3.96694435e-04, 4.13389760e-04, 4.41241835e-04,
       5.20797505e-04, 6.10202318e-04, 6.68037275e-04, 7.32506509e-04,
       7.93701038e-04, 8.53195263e-04, 9.09500406e-04, 9.46514192e-04,
       2.32592269e-04, 2.71835335e-04, 3.08167830e-04, 3.44223663e-04,
       3.79800709e-04, 4.20017983e-04, 4.63830249e-04, 5.07036340e-04,
       5.73535392e-04, 6.31669944e-04, 6.62838283e-04, 7.10128399e-04,
       7.75139430e-04, 8.65899900e-04, 9.70544817e-04, 1.06985983e-03,
       3.41864652e-04, 3.69071873e-04, 3.96323914e-04, 4.22754820e-04,
       4.46886756e-04, 4.71655425e-04, 4.98428126e-04, 5.32940670e-04,
       6.06888614e-04, 6.70523092e-04, 7.07189436e-04, 7.71613559e-04,
       8.81720975e-04, 1.09840697e-03, 1.38371368e-03, 1.53473706e-03,
       4.17014409e-04, 4.30032291e-04, 4.34701447e-04, 4.33250068e-04,
       4.38496791e-04, 4.55915550e-04, 4.31207794e-04, 4.22994781e-04,
       4.37038223e-04, 4.48761188e-04, 4.56356967e-04, 4.66349302e-04,
       4.80149902e-04, 4.99643327e-04, 5.27186319e-04, 5.57484163e-04,
       2.16507342e-05, 2.14800675e-05, 2.42409842e-05, 2.30929109e-05,
       2.50050962e-05, 2.49029163e-05, 2.54020069e-05, 2.31895119e-05,
       2.51217079e-05, 2.50334833e-05, 2.48085526e-05, 2.50862649e-05,
       2.36565447e-05, 2.40919053e-05, 2.22349281e-05, 2.54304305e-05,
       4.07712301e-04, 5.41551271e-04, 6.77760807e-04, 8.20003042e-04,
       9.50566900e-04, 1.05036958e-03, 1.11506274e-03, 1.15274324e-03,
       1.17376540e-03, 1.18031248e-03, 1.18122390e-03, 1.18179375e-03,
       1.18207792e-03, 1.18228467e-03, 1.18242903e-03, 1.18247792e-03,
       1.02293247e-03, 1.05753809e-03, 1.11542153e-03, 1.18556281e-03,
       1.24850264e-03, 1.29191973e-03, 1.30981358e-03, 1.31571468e-03,
       1.31824624e-03, 1.31927885e-03, 1.31954963e-03, 1.31999550e-03,
       1.32057443e-03, 1.32077874e-03, 1.32089714e-03, 1.32086105e-03])
ysigma_sw_abs = np.repeat(0.00065187697,224)
ymeans_sw_ray = np.array([0.00016408, 0.00016821, 0.00016852, 0.00016616, 0.0001631 ,
       0.0001615 , 0.00016211, 0.00016632, 0.00017432, 0.00017609,
       0.00017617, 0.00017683, 0.00017806, 0.00017891, 0.00017938,
       0.00017905, 0.00020313, 0.000203  , 0.00020417, 0.00020546,
       0.00020597, 0.00020647, 0.0002067 , 0.0002069 , 0.00020719,
       0.00020752, 0.00020766, 0.00020783, 0.00020801, 0.00020828,
       0.00020884, 0.00020988, 0.00022147, 0.00022575, 0.00022777,
       0.00022846, 0.00022824, 0.00022803, 0.00022816, 0.00022841,
       0.0002286 , 0.00022876, 0.00022883, 0.00022887, 0.00022888,
       0.00022891, 0.00022922, 0.00023004, 0.00025017, 0.00024942,
       0.00024824, 0.00024734, 0.00024655, 0.00024587, 0.00024539,
       0.00024486, 0.00024454, 0.0002441 , 0.00024381, 0.00024343,
       0.00024307, 0.00024265, 0.00024179, 0.00024042, 0.00025942,
       0.00026145, 0.00026296, 0.00026379, 0.00026454, 0.00026518,
       0.00026548, 0.00026566, 0.00026578, 0.00026592, 0.00026607,
       0.00026617, 0.00026612, 0.00026633, 0.00026634, 0.00026667,
       0.00028838, 0.00028652, 0.00028487, 0.00028311, 0.00027978,
       0.00027901, 0.00027885, 0.00027866, 0.000278  , 0.00027733,
       0.00027734, 0.00027694, 0.00027574, 0.00027526, 0.0002754 ,
       0.00027594, 0.00030356, 0.00031213, 0.00031563, 0.00031476,
       0.00031548, 0.00031693, 0.00031758, 0.00031764, 0.00031757,
       0.00031747, 0.0003175 , 0.0003176 , 0.00031767, 0.00031787,
       0.00031921, 0.00032022, 0.00033161, 0.00033401, 0.00033486,
       0.0003345 , 0.00033421, 0.00033408, 0.00033402, 0.00033377,
       0.00033366, 0.00033365, 0.0003337 , 0.0003337 , 0.00033372,
       0.00033368, 0.0003336 , 0.00033342, 0.00038102, 0.00038465,
       0.00038598, 0.00038884, 0.00039175, 0.00039174, 0.00039248,
       0.00039248, 0.00039339, 0.00038445, 0.00037872, 0.00037472,
       0.00037253, 0.00036779, 0.00036238, 0.00035383, 0.0004347 ,
       0.00044431, 0.00045121, 0.00045676, 0.00046053, 0.00046292,
       0.00046443, 0.00046354, 0.00045597, 0.0004476 , 0.00044526,
       0.00044227, 0.00043994, 0.00043726, 0.00043139, 0.00043131,
       0.00050991, 0.0005198 , 0.00052429, 0.00052806, 0.00052854,
       0.00052917, 0.00053388, 0.00053857, 0.0005396 , 0.00053686,
       0.00053488, 0.0005327 , 0.00052904, 0.00052511, 0.00052043,
       0.00051689, 0.00057637, 0.0005886 , 0.0006002 , 0.00061095,
       0.00062064, 0.00062908, 0.00063611, 0.00064162, 0.00064557,
       0.00064733, 0.00064765, 0.0006479 , 0.0006481 , 0.00064824,
       0.00064831, 0.00064833, 0.00065669, 0.00067188, 0.00068705,
       0.00070096, 0.00071349, 0.00072444, 0.00073358, 0.00074076,
       0.00074614, 0.00074835, 0.00074795, 0.00074786, 0.0007479 ,
       0.00074804, 0.00074818, 0.00074823, 0.0008143 , 0.00081451,
       0.00081412, 0.00081407, 0.00081408, 0.00081299, 0.00081158,
       0.00081498, 0.00081472, 0.0008145 , 0.00081539, 0.00081426,
       0.00081398, 0.000814  , 0.00081404, 0.00081415])
ysigma_sw_ray = np.repeat(0.00019679657,224)


    
def load_radscheme_rnn(fname, nneur, predictand='rsu_rsd', scale_p_h2o_o3=True, \
                                return_p=False, return_coldry=False,
                                hws_option_1=False,
                                hws_option_2=False):
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

    vmr_h2o = x_gas[:,:,2].reshape(ns,nlay)

    # Mu0 and surface albedo are also required as inputs
    # Don't know how to add constant (sequence-independent) variables,
    # so will add them as input to each sequence/level - unelegant but should work..
    mass_coord = np.copy(mu0)
    mass_coord = 1.0 / mass_coord
    mass_coord = np.repeat(mass_coord.reshape(ns,1,1),nlay,axis=1)
    mu0     = np.repeat(mu0.reshape(ns,1,1),nlay,axis=1)
    sfc_alb = np.repeat(sfc_alb.reshape(ns,1,1),nlay,axis=1)


    zero_alb = np.zeros((ns,1), dtype=np.float32)
    initial_state = np.ones((ns,nneur), dtype=np.float32)

    pres = dat.variables['pres_level'][:,:,:].data       # (nexp,ncol, nlev)
    pres = np.reshape(pres,(ns,nlev))

    pres_copy = np.copy(pres)
    pres_copy = np.reshape(pres_copy,(ns,nlev,1))

    for ii in range(nlay):
        mass_coord[:,ii,:] = (pres_copy[:,ii+1,:] - pres_copy[:,ii,:]) * 0.00001

    if scale_p_h2o_o3:
        # Log-scale pressure, power-scale H2O and O3
        x_gas[:,:,1] = np.log(x_gas[:,:,1])
        x_gas[:,:,2] = x_gas[:,:,2]**(1.0/4) 
        x_gas[:,:,3] = x_gas[:,:,3]**(1.0/4)
    
    rsu_raw = np.copy(rsu)
    rsd_raw = np.copy(rsd)
    
    # normalize downwelling flux by the boundary condition
    if hws_option_2:
        rsd0    = np.full((rsd.shape[0],), 1412.0)
    else:
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

    top_output = np.ones((ns,1),dtype=np.float32)

    # Concatenate inputs and outputs...
    if hws_option_1:
        # substitue mass coord for albedo
        x       = np.concatenate((x_gas,lwp,iwp,mu0,sfc_alb,mass_coord),axis=2)
    else:
        x       = np.concatenate((x_gas,lwp,iwp,mu0,sfc_alb),axis=2)
    y       = np.concatenate((rsd,rsu),axis=2)

    print( "there are {} profiles in this dataset ({} experiments, {} columns)".format(nexp*ncol,nexp,ncol))
    
    #if return_coldry:
    #    coldry = get_col_dry(vmr_h2o,pres)
    
    dat.close()
    if return_p:
        if return_coldry:
            return x,y,rsd0,rsu0,rsd_raw,rsu_raw,pres,coldry, zero_alb, initial_state, top_output
        else:
            return x,y,rsd0,rsu0,rsd_raw,rsu_raw,pres, zero_alb, initial_state, top_output
    
    else:
        if return_coldry:
            return x,y,rsd0,rsu0,rsd_raw,rsu_raw, coldry, zero_alb, initial_state, top_output
        else:
            return x,y,rsd0,rsu0,rsd_raw,rsu_raw, zero_alb, initial_state, top_output

def load_radscheme_rnn_direct(fname, predictand='rsu_rsd', scale_p_h2o_o3=True, \
                                return_p=False, return_coldry=False,
                                hws_option_1=False,
                                hws_option_2=False,
                                use_mass_coord=False):
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
    rsd = dat.variables['rsd_dir'][:]
        
    if np.size(rsu.shape) != 3:
        sys.exit("Invalid array shapes, RTE output should have 3 dimensions")
    
    # Reshape to profiles...
    x_gas   = np.reshape(x_gas,(ns,nlay,nx)) 
    lwp     = np.reshape(lwp,  (ns,nlay,1))    
    iwp     = np.reshape(iwp,  (ns,nlay,1))
    rsu     = np.reshape(rsu,  (ns,nlev))
    rsd     = np.reshape(rsd,  (ns,nlev))

    top_output = np.ones((ns,1),dtype=np.float32)

    vmr_h2o = np.copy(x_gas[:,:,2].reshape(ns,nlay))

    # Mu0 and surface albedo are also required as inputs
    # Don't know how to add constant (sequence-independent) variables,
    # so will add them as input to each sequence/level - unelegant but should work..

    mu0     = np.repeat(mu0.reshape(ns,1,1),nlay,axis=1)
    sfc_alb = np.repeat(sfc_alb.reshape(ns,1,1),nlay,axis=1)

    pres = dat.variables['pres_level'][:,:,:].data       # (nexp,ncol, nlev)
    pres = np.reshape(pres,(ns,nlev))

    pres_copy = np.copy(pres)
    pres_copy = np.reshape(pres_copy,(ns,nlev,1))

    m_dry = 2.8964
    m_h2o =  1.8016
    factor = 0.0001 / (m_dry + m_h2o * vmr_h2o)

    if use_mass_coord:
        mass_coord = (pres_copy[:,1:,:] - pres_copy[:,:-1,:]) * factor
        x_gas[:,:,2] = x_gas[:,:,2] * mass_coord
        x_gas[:,:,3] = x_gas[:,:,3] * mass_coord
        x_gas[:,:,4] = x_gas[:,:,4] * mass_coord
        x_gas[:,:,5] = x_gas[:,:,5] * mass_coord
        x_gas[:,:,6] = x_gas[:,:,6] * mass_coord

    if scale_p_h2o_o3:
        # Log-scale pressure, power-scale H2O and O3
        x_gas[:,:,1] = np.log(x_gas[:,:,1])
        x_gas[:,:,2] = x_gas[:,:,2]**(1.0/4) 
        x_gas[:,:,3] = x_gas[:,:,3]**(1.0/4)
    
    rsu_raw = np.copy(rsu)
    rsd_raw = np.copy(rsd)
    
    # normalize downwelling flux by the boundary condition
    if hws_option_2:
        rsd0    = np.full((rsd.shape[0],), 1412.0)
    else:
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



    # Concatenate inputs and outputs...
    if hws_option_1:
        # substitue mass coord for albedo
        x       = np.concatenate((x_gas,lwp,iwp,mu0,sfc_alb,mass_coord),axis=2)
    else:
        x       = np.concatenate((x_gas,lwp,iwp,mu0,sfc_alb),axis=2)
    y       = rsd #np.concatenate((rsd,rsu),axis=2)

    print( "there are {} profiles in this dataset ({} experiments, {} columns)".format(nexp*ncol,nexp,ncol))
    
    #if return_coldry:
    #    coldry = get_col_dry(vmr_h2o,pres)
    
    dat.close()
    if return_p:
        if return_coldry:
            return x,y,rsd0,rsu0,rsd_raw,rsu_raw,pres,coldry, top_output
        else:
            return x,y,rsd0,rsu0,rsd_raw,rsu_raw,pres, top_output
    
    else:
        if return_coldry:
            return x,y,rsd0,rsu0,rsd_raw,rsu_raw, coldry, top_output
        else:
            return x,y,rsd0,rsu0,rsd_raw,rsu_raw, top_output

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
