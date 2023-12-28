"""
Modified by Henry Schneiderman
- Added code to properly reverse output from intermediate RNN
- Added functionality to work with Dataset allowing a list of input
files

Python framework for developing neural networks to replace radiative
transfer computations, either fully or just one component

This code is for emulating RTE+RRTMGP (entire radiation scheme)

This program takes existing input-output data generated with RTE+RRTMGP and
user-specified hyperparameters such as the number of neurons, optionally
scales the data, and trains a neural network. 

Temporary code

Contributions welcome!

@author: Peter Ukkonen
"""
import os
import gc
import numpy as np
import time

import matplotlib.pyplot as plt

import ml_loaddata_rnn
import ml_data_generator
def calc_heatingrate(F, p):
    dF = F[:,1:] - F[:,0:-1] 
    dp = p[:,1:] - p[:,0:-1] 
    dFdp = dF/dp
    g = 9.81 # m s-2
    cp = 1004 # J K-1  kg-1
    dTdt = -(g/cp)*(dFdp) # K / s
    dTdt_day = (24*3600)*dTdt
    return dTdt_day

def rmse(predictions, targets,ax=0):
    return np.sqrt(((predictions - targets) ** 2).mean(axis=ax))

def mse(predictions, targets,ax=0):
    return ((predictions - targets) ** 2).mean(axis=ax)

def mae(predictions,targets,ax=0):
    diff = predictions - targets
    return np.mean(np.abs(diff),axis=ax)

def mape(a,b):
    mask = a != 0
    return 100*(np.fabs(a[mask] - b[mask])/a[mask]).mean()

def plot_flux_and_hr_error(rsu_true, rsd_true, rsu_pred, rsd_pred, pres):
    pres_lay = 0.5 * (pres[:,1:] + pres[:,0:-1])

    dTdt_true = calc_heatingrate(rsd_true - rsu_true, pres)
    dTdt_pred = calc_heatingrate(rsd_pred - rsu_pred, pres)
    
    bias_tot = np.mean(dTdt_pred.flatten()-dTdt_true.flatten())
    rmse_tot = rmse(dTdt_true.flatten(), dTdt_pred.flatten())
    mae_tot = mae(dTdt_true.flatten(), dTdt_pred.flatten())
    mae_percent = 100 * np.abs(mae_tot / dTdt_true.mean())
    # mae_percent = mape(dTdt_true.flatten(), dTdt_pred.flatten())
    r2 =  np.corrcoef(dTdt_pred.flatten(),dTdt_true.flatten())[0,1]; r2 = r2**2
    str_hre = 'Heating rate error \nR$^2$: {:0.4f} \nBias: {:0.3f} \nRMSE: {:0.3f} \nMAE: {:0.3f} ({:0.1f}%)'.format(r2,
                                    bias_tot,rmse_tot, mae_tot, mae_percent)
    mae_rsu = mae(rsu_true.flatten(), rsu_pred.flatten())
    mae_rsd = mae(rsd_true.flatten(), rsd_pred.flatten())
    mae_rsu_p = 100 * np.abs(mae_rsu / rsu_true.mean())
    mae_rsd_p = 100 * np.abs(mae_rsd / rsd_true.mean())

    str_rsu =  'Upwelling flux error \nMAE: {:0.2f} ({:0.1f}%)'.format(mae_rsu, mae_rsu_p)
    str_rsd =  'Downwelling flux error \nMAE: {:0.2f} ({:0.1f}%)'.format(mae_rsd, mae_rsd_p)
    errfunc = mae
    #errfunc = rmse
    ind_p = 0
    hr_err      = errfunc(dTdt_true[:,ind_p:], dTdt_pred[:,ind_p:])
    fluxup_err  = errfunc(rsu_true[:,ind_p:], rsu_pred[:,ind_p:])
    fluxdn_err  = errfunc(rsd_true[:,ind_p:], rsd_pred[:,ind_p:])
    fluxnet_err  = errfunc((rsd_true[:,ind_p:] - rsu_true[:,ind_p:]), \
                           (rsd_pred[:,ind_p:] - rsu_pred[:,ind_p:] ))
    yy = 0.01*pres[:,:].mean(axis=0)
    yy2 = 0.01*pres_lay[:,:].mean(axis=0)

    fig, (ax0,ax1) = plt.subplots(ncols=2, sharey=True)
    ax0.plot(hr_err,  yy2[ind_p:], label=str_hre)
    ax0.invert_yaxis()
    ax0.set_ylabel('Pressure (hPa)',fontsize=15)
    ax0.set_xlabel('Heating rate (K h$^{-1}$)',fontsize=15); 
    ax1.set_xlabel('Flux (W m$^{-2}$)',fontsize=15); 
    ax1.plot(fluxup_err,  yy[ind_p:], label=str_rsu)
    ax1.plot(fluxdn_err,  yy[ind_p:], label=str_rsd)
    ax1.plot(fluxnet_err,  yy[ind_p:], label='Net flux error')
    ax0.legend(); ax1.legend()
    ax0.grid(); ax1.grid()

# ----------------------------------------------------------------------------
# ----------------- RTE+RRTMGP EMULATION  ------------------------
# ----------------------------------------------------------------------------

#  ----------------- File paths -----------------

datadir     = "/data-T1/hws/tmp/"
fpath       = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"
fpath_val   = datadir + "/RADSCHEME_data_g224_CAMS_2014.2.nc"
fpath_test  = datadir +  "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"
# fpath_test  = datadir +  "/RADSCHEME_data_g224_NWPSAFtest.nc"

# ----------- config ------------

scale_inputs    = True

# Model training and evaluation: use GPU or CPU?
use_gpu = True


model_name = 'MODEL.RNN_3_Dataset.'
is_train = False

weight_prof = ml_loaddata_rnn.get_weight_profile (fpath)

import tensorflow as tf
from tensorflow.keras import losses, optimizers, layers, Input, Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense,TimeDistributed

mymetrics   = ['mean_absolute_error']
valfunc     = 'val_mean_absolute_error'


def my_gradient_tf(a):
    return a[:,1:] - a[:,0:-1]

def calc_heatingrates_tf_dp(flux_dn, flux_up, dp):
    #  flux_net =   flux_up   - flux_dn
    F = tf.subtract(flux_up, flux_dn)
    dF = my_gradient_tf(F)
    dFdp = tf.divide(dF, dp)
    coeff = -844.2071713147411#  -(24*3600) * (9.81/1004)  
    dTdt_day = tf.multiply(coeff, dFdp)
    return dTdt_day


def CustomLoss(y_true, y_pred, dp, rsd_top):
    # err_flux =  tf.math.reduce_mean(tf.math.square(weight_prof*(y_true - y_pred)),axis=-1)
    err_flux =  K.mean(K.square(weight_prof*(y_true - y_pred)))
    # err_flux =  K.mean(K.square((y_true - y_pred)))

    rsd_true = tf.math.multiply(y_true[:,:,0], rsd_top)
    rsd_pred = tf.math.multiply(y_pred[:,:,0], rsd_top)
    rsu_true = tf.math.multiply(y_true[:,:,1], rsd_top)
    rsu_pred = tf.math.multiply(y_pred[:,:,1], rsd_top)

    HR_true = calc_heatingrates_tf_dp(rsd_true, rsu_true, dp)
    HR_pred = calc_heatingrates_tf_dp(rsd_pred, rsu_pred, dp)
    # err_hr = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(HR_true - HR_pred),axis=-1))
    err_hr = K.sqrt(K.mean(K.square(HR_true - HR_pred)))

    # alpha   = 1e-6
    # alpha   = 1e-5
    alpha   = 1e-4
    return (alpha) * err_hr + (1 - alpha)*err_flux   


def rmse_hr(y_true, y_pred, dp, rsd_top):
    
    rsd_true = tf.math.multiply(y_true[:,:,0], rsd_top)
    rsd_pred = tf.math.multiply(y_pred[:,:,0], rsd_top)
    rsu_true = tf.math.multiply(y_true[:,:,1], rsd_top)
    rsu_pred = tf.math.multiply(y_pred[:,:,1], rsd_top)

    HR_true = calc_heatingrates_tf_dp(rsd_true, rsu_true, dp)
    HR_pred = calc_heatingrates_tf_dp(rsd_pred, rsu_pred, dp)

    # return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(HR_true - HR_pred),axis=-1))
    return K.sqrt(K.mean(K.square(HR_true - HR_pred)))


def rmse_flux(y_true, y_pred, dp, rsd_top):

    rsd_top_2 = tf.expand_dims(rsd_top, axis=2)
    y_true_scaled = tf.math.multiply(y_true, rsd_top_2)
    y_pred_scaled = tf.math.multiply(y_pred, rsd_top_2)

    return K.sqrt(K.mean(K.square(y_true_scaled - y_pred_scaled)))

# MODEL TRAINING CODE
if True:
    
    # Model architecture
    # Activation for first
    # activ0 = 'relu'
    activ0 = 'linear'
    # activ0 = 'softsign'
    # activ0 = 'tanh'
    activ_last   = 'relu'
    activ_last   = 'sigmoid'
    # activ_last   = 'linear'
    
    epochs      = 100000
    patience    = 400 #25
    lossfunc    = losses.mean_squared_error
    batch_size  = 1024
    batch_size  = 2048

    nneur = 16  
    nlay = 60
    nx_main = 10 # 5-gases + temp + pressure + mu + lwp + iwp
    nx_aux = 1
    ny = 2  # rsd and rsu

    lr = 0.001 # DEFAULT!
    optim = optimizers.Adam(learning_rate=lr)
    
    # shape(x_tr_m) = (nsamples, nseq, nfeatures_seq)
    # shape(x_tr_s) = (nsamples, nfeatures_aux)
    # shape(y_tr)   = (nsamples, nseq, noutputs)
    
    # Main inputs associated with RNN layer (sequence dependent)
    # inputs = Input(shape=(None,nx_main),name='inputs_main')
    inputs = Input(shape=(nlay,nx_main),name='inputs_main')
    
    # Optionally, use auxiliary inputs that do not dependend on sequence?
    
    # if use_auxinputs:  # commented out cos I need albedo for the loss function, easier to have it separate
    inp_aux_albedo = Input(shape=(nx_aux),name='inputs_aux_albedo') # sfc_albedo
    
    # Target outputs: these are fed as part of the input to avoid problem with 
    # validation data where TF complained about wrong shape
    # target  = Input((None,ny))
    target  = Input((nlay+1,ny))
    
    # other inputs required to compute heating rate
    dpres   = Input((nlay,))
    incflux = Input((nlay+1))
    
    # hidden0,last_state = layers.SimpleRNN(nneur,return_sequences=True,return_state=True)(inputs)
    hidden0,last_state = layers.GRU(nneur,return_sequences=True,return_state=True)(inputs)
    
    last_state_plus_albedo =  tf.concat([last_state,inp_aux_albedo],axis=1)
    
    mlp_surface_outp = Dense(nneur, activation=activ0,name='dense_surface')(last_state_plus_albedo)
    
    hidden0_lev = tf.concat([hidden0,tf.reshape(mlp_surface_outp,[-1,1,nneur])],axis=1)
    
    hidden1 = layers.GRU(nneur,return_sequences=True,go_backwards=True)(hidden0_lev)
    # hidden1 = layers.SimpleRNN(nneur,return_sequences=True,go_backwards=True)(hidden0_lev)
    
    ###### Only processing change from original ######
    hidden1 = tf.reverse(hidden1, [1])
    ##################################################

    # try concatinating hidden0 and hidden 1 instead
    hidden_concat  = tf.concat([hidden0_lev,hidden1],axis=2)

    hidden2 = layers.GRU(nneur,return_sequences=True)(hidden_concat)
    # hidden2 = layers.SimpleRNN(nneur,return_sequences=True)(hidden_concat)

    outputs = TimeDistributed(layers.Dense(ny, activation=activ_last),name='dense_output')(hidden2)

    model = Model(inputs=[inputs, inp_aux_albedo, target, dpres, incflux], outputs=outputs)
    
    model.add_metric(rmse_flux(target,outputs,dpres,incflux),'rmse_flux')
    model.add_metric(rmse_hr(target,outputs,dpres,incflux),'rmse_hr')

    # model.add_metric(rmse_hr(target,outputs,inp_aux_albedo,dpres,incflux),'rmse_hr')
    # model.add_metric(rmse_flux(target,outputs,inp_aux,incflux),'rmse_flux')
    
    model.add_loss(CustomLoss(target,outputs,dpres, incflux))
    # model.add_loss(losses.mean_squared_error(target,outputs))
    model.compile(optimizer=optim,loss='mse')
    
    model.summary()

    callbacks = [EarlyStopping(monitor='rmse_hr',  patience=patience, verbose=1, \
                                 mode='min',restore_best_weights=True)]
    
    epoch_period = 25
    n_epochs = 0  
    #steps_per_epoch = 924

    train_input_dir = "/data-T1/hws/CAMS/processed_data/training/2008/"
    cross_input_dir = "/data-T1/hws/CAMS/processed_data/cross_validation/2008/"
    months = [str(m).zfill(2) for m in range(1,13)]
    train_input_files = [f'{train_input_dir}Flux_sw-2008-{month}.nc' for month in months]
    cross_input_files = [f'{cross_input_dir}Flux_sw-2008-{month}.nc' for month in months]

    generator_training = ml_data_generator.InputSequence(train_input_files, batch_size)
    #generator_training = generator_training.batch(batch_size)
    generator_cross_validation = ml_data_generator.InputSequence(cross_input_files, batch_size)

    #generator_cross_validation = generator_cross_validation.batch(batch_size)

    if n_epochs > 0:
        del model
        model = tf.keras.models.load_model(datadir + model_name + str(n_epochs))

    if is_train:
        while n_epochs < epochs:

            history = model.fit(generator_training, \
            epochs=epoch_period, \
            shuffle=False, verbose=1,  \
            validation_data=generator_cross_validation, callbacks=callbacks)

            #history = model.fit_generator(generator_training, \
            #steps_per_epoch, epochs=epoch_period,   \
            #validation_data=generator_cross_validation, #callbacks=callbacks)

            #print(history.history.keys())
            #print("number of epochs = " + str(history.history['epoch']))
            print(str(history.history['rmse_hr']))

            nn_epochs = len(history.history['rmse_hr'])
            if  nn_epochs < epoch_period:
                model.save(datadir + model_name + str(n_epochs + nn_epochs))
                print("Writing FINAL model. N_epochs = " + str(n_epochs + nn_epochs))
                break
            else:
                n_epochs = n_epochs + epoch_period
                model.save(datadir + model_name + str(n_epochs))
                print(f"Writing model (n_epochs = {n_epochs})")

        del model
        model = tf.keras.models.load_model(datadir + model_name + str(n_epochs))

    else:
        year = '2009'
        testing_input_dir = f"/data-T1/hws/CAMS/processed_data/testing/{year}/"
        testing_input_files = [f'{testing_input_dir}Flux_sw-{year}-{month}.2.nc' for month in months]
        n_epochs = 800
        generator_testing = ml_data_generator.InputSequence(testing_input_files, batch_size)
        while n_epochs < epochs:
            n_epochs = n_epochs + epoch_period
            print(f"n_epochs = {n_epochs}")
            model = tf.keras.models.load_model(datadir + model_name + str(n_epochs))
            model.evaluate(x=generator_testing)
            
    