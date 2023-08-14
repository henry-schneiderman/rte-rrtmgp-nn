#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted from ml_train_radscheme_brnn2.py by Henry Schneiderman

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

from ml_loaddata_hws import load_radscheme_rnn_direct, preproc_divbymax

import matplotlib.pyplot as plt

        

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
datadir     = "/media/peter/samsung/data/CAMS/ml_training/"
datadir     = "/home/peter/data/"
datadir     = "/home/hws/tmp/"
fpath       = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"
fpath_val   = datadir + "/RADSCHEME_data_g224_CAMS_2014.2.nc"
fpath_test  = datadir +  "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"
# fpath_test  = datadir +  "/RADSCHEME_data_g224_NWPSAFtest.nc"

# ----------- config ------------

scale_inputs    = True

# didn't seem to improve results
# include_deltap = True


# do outputs consist of the entire profile, i.e. fluxes on halv levels (nlay+1)
# otherwise the TOA incoming flux and surface downward flux can be omitted for nlay outputs
train_on_levflux = True


# Model training and evaluation: use GPU or CPU?
use_gpu = False
# test inference of existing model 
evaluate_onnx = True
final_evaluation = False

hws_option_1 = False
hws_option_2 = False
hws_option_3 = False #True

reverse_sequence = True

# ----------- config ------------

# Load data

x_tr_raw, y_tr_raw, rsd0_tr, rsu0_tr, rsd_tr, rsu_tr, pres_tr, top_output_tr= \
    load_radscheme_rnn_direct(fpath,  scale_p_h2o_o3 = scale_inputs, return_p=True, 
    hws_option_1=hws_option_1,
    hws_option_2=hws_option_2)
    
x_val_raw, y_val_raw, rsd0_val, rsu0_val,rsd_val,rsu_val,  pres_val, top_output_val = \
    load_radscheme_rnn_direct(fpath_val, scale_p_h2o_o3 = scale_inputs, return_p=True,
    hws_option_1=hws_option_1,
    hws_option_2=hws_option_2) 

x_test_raw, y_test_raw, rsd0_test, rsu0_test, rsd_test, rsu_test, pres_test, top_output_test = \
    load_radscheme_rnn_direct(fpath_test,  scale_p_h2o_o3 = scale_inputs, return_p=True, 
    hws_option_1=hws_option_1,
    hws_option_2=hws_option_2)

if scale_inputs:
    x_tr        = np.copy(x_tr_raw)
    x_val       = np.copy(x_val_raw)
    x_test      = np.copy(x_test_raw)
    # x[0:7] gas concentrations, x[7] lwp, x[8] iwp, x[9] mu0, x[10] sfc_alb
    if hws_option_1:
        xmax = np.array([3.20104980e+02, 1.15657101e+01, 4.46762830e-01, 5.68951890e-02,
           6.59545418e-04, 3.38450207e-07, 5.08802714e-06, 2.13372910e+02,
           1.96923096e+02, 1.00000000e+00, 1.00, 1.0],dtype=np.float32)
    else:
        xmax = np.array([3.20104980e+02, 1.15657101e+01, 4.46762830e-01, 5.68951890e-02,
           6.59545418e-04, 3.38450207e-07, 5.08802714e-06, 2.13372910e+02,
           1.96923096e+02, 1.00000000e+00, 1.00],dtype=np.float32)
    # We do not scale albedo values to (0..1) because its already in the 
    # approximate range and we need the true albedo value for postprocessing.
    # the max for x_tr[:,:,index_albedo] is around 0.85
    x_tr            = preproc_divbymax(x_tr_raw, xmax)
    # x_tr, xmax      = preproc_divbymax(x_tr_raw)
    x_val           = preproc_divbymax(x_val_raw, xmax)
    x_test          = preproc_divbymax(x_test_raw, xmax)
else:
    x_tr    = x_tr_raw
    x_val   = x_val_raw
    x_test  = x_test_raw
  
del x_tr_raw
    
# Number of inputs and outputs    
nx = x_tr.shape[-1]
ny = y_tr_raw.shape[-1]   
nlay = x_tr.shape[-2]
  
if train_on_levflux:
    y_tr = np.expand_dims(rsd_tr,2)
    y_val = np.expand_dims(rsd_val,2)
    y_test = np.expand_dims(rsd_test,2)
    del y_tr_raw, y_val_raw, y_test_raw
    for i in range(1):
        for j in range(61):
            y_tr[:,j,i] = y_tr[:,j,i] / rsd0_tr
            y_val[:,j,i] = y_val[:,j,i] / rsd0_val
            y_test[:,j,i] = y_test[:,j,i] / rsd0_test
else:
    y_tr = y_tr_raw; y_val = y_val_raw; y_test = y_test_raw

rsd0_tr_big     = rsd0_tr.reshape(-1,1).repeat(nlay+1,axis=1)
rsd0_val_big    = rsd0_val.reshape(-1,1).repeat(nlay+1,axis=1)
rsd0_test_big   = rsd0_test.reshape(-1,1).repeat(nlay+1,axis=1)

x_tr_m = x_tr[:,:,0:-1];  x_val_m = x_val[:,:,0:-1]
x_test_m = x_test[:,:,0:-1]
x_tr_aux1 = x_tr[:,0,-1:]; x_val_aux1 = x_val[:,0,-1:]; 
x_test_aux1 = x_test[:,0,-1:]    
nx_aux = 1

nx_main = x_tr_m.shape[-1]
    
hre_loss = True
if hre_loss:
    # dp_tr = np.gradient(pres_tr,axis=1)
    # dp_val = np.gradient(pres_val,axis=1)
    dp_tr    = pres_tr[:,1:] - pres_tr[:,0:-1] 
    dp_val   = pres_val[:,1:] - pres_val[:,0:-1] 
    dp_test  = pres_test[:,1:] - pres_test[:,0:-1] 


# rsd_mean_tr = y_tr[:,:,0].mean(axis=0)
# rsu_mean_tr = y_tr[:,:,1].mean(axis=0)
# yy = 0.01*pres_tr[:,1:].mean(axis=0)
# fig, ax = plt.subplots()
# ax.plot(rsd_mean_tr,  yy, label='RSD')
# ax.plot(rsu_mean_tr,  yy, label='RSU')
# ax.invert_yaxis()
# ax.set_ylabel('Pressure (hPa)',fontsize=15)
# in the loss function, test a weight profile to normalize 
# everything to 1 (so that different levels have equal contributions)
weight_prof = 1/y_tr.mean(axis=0)
# ax.plot(weight_prof[:,0],yy,label='RSD weight')
# ax.plot(weight_prof[:,1],yy,label='RSU weight')
# ax.legend(); ax.grid()

# Ready for training


import tensorflow as tf
from tensorflow.keras import losses, optimizers, layers, Input, Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense,TimeDistributed

mymetrics   = ['mean_absolute_error']
valfunc     = 'val_mean_absolute_error'

# if use_gpu:
#     devstr = '/gpu:0'
#     # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# else:
#     num_cpu_threads = 12
#     devstr = '/cpu:0'
#     # Maximum number of threads to use for OpenMP parallel regions.
#     os.environ["OMP_NUM_THREADS"] = str(num_cpu_threads)
#     # Without setting below 2 environment variables, it didn't work for me. Thanks to @cjw85 
#     os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_cpu_threads)
#     os.environ["TF_NUM_INTEROP_THREADS"] = str(1)
#     os.environ['KMP_BLOCKTIME'] = '1' 

#     tf.config.threading.set_intra_op_parallelism_threads(
#         num_cpu_threads
#     )
#     tf.config.threading.set_inter_op_parallelism_threads(
#         1
#     )
#     tf.config.set_soft_device_placement(True)
#     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def my_gradient_tf(a):
    return a[:,1:] - a[:,0:-1]


def calc_heatingrates_tf_dp(flux_dn, dp):
    #  flux_net =   flux_up   - flux_dn
    dF = my_gradient_tf(-flux_dn)
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

    HR_true = calc_heatingrates_tf_dp(rsd_true, dp)
    HR_pred = calc_heatingrates_tf_dp(rsd_pred, dp)
    # err_hr = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(HR_true - HR_pred),axis=-1))
    err_hr = K.sqrt(K.mean(K.square(HR_true - HR_pred)))

    # alpha   = 1e-6
    # alpha   = 1e-5
    alpha   = 1e-4
    return (alpha) * err_hr + (1 - alpha)*err_flux   


def rmse_hr(y_true, y_pred, dp, rsd_top):
    
    rsd_true = tf.math.multiply(y_true[:,:,0], rsd_top)
    rsd_pred = tf.math.multiply(y_pred[:,:,0], rsd_top)

    HR_true = calc_heatingrates_tf_dp(rsd_true, dp)
    HR_pred = calc_heatingrates_tf_dp(rsd_pred, dp)

    # return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(HR_true - HR_pred),axis=-1))
    return K.sqrt(K.mean(K.square(HR_true - HR_pred)))

# def my_sigmoid(x):
#     x = tf.keras.activations.sigmoid(x)
#     xshape = tf.shape(x) # = (batch_size, 61, 2)
#     xones = tf.ones(xshape)

#     boolmask = tf.expand_dims(boolmask,2)
#     boolmask = tf.transpose(boolmask, perm=[2, 1,0])
#     boolmask = tf.repeat(boolmask, xshape[0],axis=0)
#     xn = tf.where(boolmask, xones, x)

#     return xn

# MODEL TRAINING CODE
if not final_evaluation:
    
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
    patience    = 1000 #25
    lossfunc    = losses.mean_squared_error
    batch_size  = 1024
    batch_size  = 2048
    
    # nneur = 64 
    nneur = 16  
    # nneur = 24
    # nneur = 32 
    # neur  = 128
    # nneur = 12
    # Input for variable-length sequences of integers
    # inputs = Input(shape=(None,nlay,nx))
    
    mergemode = 'concat'
    # # mergemode = 'sum' #worse
    # mergemode = 'ave' # better
    # mergemode = 'mul' # best?
    
    # lr          = 0.0001 
    # lr          = 0.0001 
    lr = 0.001 # DEFAULT!
    optim = optimizers.Adam(learning_rate=lr)
    
    # shape(x_tr_m) = (nsamples, nseq, nfeatures_seq)
    # shape(x_tr_s) = (nsamples, nfeatures_aux)
    # shape(y_tr)   = (nsamples, nseq, noutputs)
    
    # Main inputs associated with RNN layer (sequence dependent)
    # inputs = Input(shape=(None,nx_main),name='inputs_main')
    inputs = Input(shape=(nlay,nx_main),batch_size=batch_size, name='inputs_main')

    print("Inputs shape: " + str(inputs.shape))
    
    # Optionally, use auxiliary inputs that do not dependend on sequence?


    
    # Target outputs: these are fed as part of the input to avoid problem with 
    # validation data where TF complained about wrong shape
    # target  = Input((None,ny))
    target  = Input(shape=(nlay+1,ny), batch_size=batch_size,name='target')
    
    # other inputs required to compute heating rate
    dpres   = Input((nlay,), batch_size=batch_size,name="dpres")
    incflux = Input((nlay+1), batch_size=batch_size,name="incflux")
    top_output = Input((1),batch_size=batch_size,name="top_output")
    
    inp_aux_albedo = Input(shape=(nx_aux),batch_size=batch_size,name='inputs_aux_albedo') 
            
    # hidden0,last_state = layers.SimpleRNN(nneur,return_sequences=True,return_state=True)(inputs)
    hidden0, last_state = layers.GRU(nneur,return_sequences=True,return_state=True)(inputs)
    
    #last_state_plus_albedo =  tf.concat([last_state,inp_aux_albedo],axis=1)

    #mlp_surface_outp = Dense(nneur, activation=activ0,name='dense_surface')(last_state_plus_albedo)

    outputs = TimeDistributed(layers.Dense(ny, activation=activ_last),name='dense_output')(hidden0)

    outputs = tf.concat([tf.reshape(top_output, (-1,1,1)),outputs], axis=1)
    
    # if use_auxinputs:
    #     if only_albedo_as_auxinput:
    #         model = Model(inputs=[inputs, inp_aux_albedo, target, dpres, incflux], outputs=outputs)
    #     else:
    #         model = Model(inputs=[inputs, inp_aux_mu, inp_aux_albedo, target, dpres, incflux], outputs=outputs)
    # else:
    #     model = Model(inputs=[inputs, target, dpres, incflux], outputs=outputs)

    model = Model(inputs=[inputs, inp_aux_albedo, target, dpres, incflux, top_output], outputs=outputs)
    
    model.add_metric(rmse_hr(target,outputs,dpres,incflux),'rmse_hr')
    
    # model.add_metric(rmse_hr(target,outputs,inp_aux_albedo,dpres,incflux),'rmse_hr')
    # model.add_metric(rmse_flux(target,outputs,inp_aux,incflux),'rmse_flux')
    
    model.add_loss(CustomLoss(target,outputs,dpres, incflux))
    # model.add_loss(losses.mean_squared_error(target,outputs))
    model.compile(optimizer=optim,loss='mse')
    
    model.summary()
    
    # model.add_metric(heatingrateloss,'heating_rate_mse')
    # model.metrics_names.append("heating_rate_mse")
    
    # # Create earlystopper and possibly other callbacks
    callbacks = [EarlyStopping(monitor='rmse_hr',  patience=patience, verbose=1, \
                                 mode='min',restore_best_weights=True)]
    
    
    epoch_period = 200
    n_epochs = 0
    fpath = 'saved_model/bigru_gru_16_nocoldry_levoutput.HWS7.'

    if False:
        n_epochs = 200
        del model
        model = tf.keras.models.load_model(fpath + 'DIRECT_TEMP.' + str(n_epochs))
    while n_epochs < epochs:

        history = model.fit(x=[x_tr_m, x_tr_aux1, y_tr, dp_tr, rsd0_tr_big, top_output_tr], y=None, \
                epochs=epoch_period, batch_size=batch_size, shuffle=True, verbose=1,  \
                validation_data=[x_val_m, x_val_aux1, y_val, dp_val, rsd0_val_big, top_output_val], callbacks=callbacks)  
        
        #print(history.history.keys())
        #print("number of epochs = " + str(history.history['epoch']))
        print(str(history.history['rmse_hr']))

        nn_epochs = len(history.history['rmse_hr'])
        if  nn_epochs < epoch_period:
            model.save(fpath + 'DIRECT_FINAL')
            print("Writing FINAL model. N_epochs = " + str(n_epochs + nn_epochs))
            break
        else:
            n_epochs = n_epochs + epoch_period
            model.save(fpath + 'DIRECT_TEMP.' + str(n_epochs))
            print("Writing model " + str(n_epochs))

        del model
        model = tf.keras.models.load_model(fpath + 'DIRECT_TEMP.' + str(n_epochs))

    # extract weights to save as simpler model without custom functions
    # layers 2,3,4,5,9 have weights
    all_weights = []
    for layer in model.layers:
      w = layer.weights
      try:      
        w[0]
        all_weights.append(w)
      except:
        pass # not all layers have weights !
    
    
    # make a new model without the functions
    # inputs = Input(shape=(None,nx_main),name='inputs_main')
    # if not only_albedo_as_auxinput: inp_aux_mu = Input(shape=(nx_aux),name='inputs_aux1') # mu0
    # inp_aux_albedo = Input(shape=(nx_aux),name='inputs_aux2') # sfc_albedo
    # if only_albedo_as_auxinput:
    #     mlp_dense_inp1 = Dense(nneur, activation=activ0,name='dense_inputs_alb')(inp_aux_albedo)
    # else:
    #     mlp_dense_inp1 = Dense(nneur, activation=activ0,name='dense_inputs_alb')(inp_aux_mu)
    # mlp_dense_inp2 = Dense(nneur, activation=activ0,name='dense_inputs_mu')(inp_aux_albedo)
    # layer_rnn = layers.GRU(nneur,return_sequences=True)
    # layer_rnn2 = layers.GRU(nneur,return_sequences=True)
    # hidden = layers.Bidirectional(layer_rnn, merge_mode=mergemode, name ='bidirectional')\
    #     (inputs, initial_state= [mlp_dense_inp1,mlp_dense_inp2])
    # hidden2 = layer_rnn2(hidden)
    # outputs = TimeDistributed(layers.Dense(ny, activation=activ_last),name='dense_output')(hidden2)




# INFERENCE USING EXISTING ONNX MODEL
# Test data should NOT be used to compare models, only once for final evaluation!

if evaluate_onnx:
    fpath = 'saved_model/bigru_gru_16_nocoldry_levoutput_FINAL'
    fpath_onnx = fpath+".onnx"
    if final_evaluation:
        xaux = x_test_aux1.reshape(-1,1)
        x_main = x_test_m
        rsd0 = rsd0_test
        rsu = rsu_test; rsd = rsd_test
        pres = pres_test
    else:
        xaux = x_val_aux1.reshape(-1,1)
        x_main = x_val_m
        rsd0 = rsd0_val
        rsu = rsu_val; rsd = rsd_val
        pres = pres_val
        
    # MODEL SAVING; first as TensorFlow SaveModel
    # fpath = 'saved_model/tmp2_bigru_gru_32'
    # fpath = 'saved_model/tmp2_bisimple_simple_32_100epochs'
    # fpath = 'saved_model/tmp'
    # fpath = 'saved_model/bigru_gru_16_nocoldry_levoutput_FINAL'
    # fpath = 'saved_model/levoutput_bigru_gru_24'
    # fpath_onnx = fpath+".onnx"
    # 
    # newmodel.save(fpath, save_format='tf')
    # newmodel.save(fpath, save_format='tf',save_traces=False)
    
    # Now convert to ONNX model
    # os.system("python -m tf2onnx.convert --saved-model {} --output {} --opset 13".format(fpath,fpath_onnx)) 
    
    if not use_gpu:
        # Force one thread on CPUs: Check environmental variables just in case
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
        import onnxruntime as ort
    else:
        import onnxruntime as ort
    
    if not use_gpu:
        # Adjust session options
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess = ort.InferenceSession(fpath_onnx, sess_options=opts)
    else:
        sess = ort.InferenceSession(fpath_onnx, providers=["CUDAExecutionProvider"])
    
    
    # sess.get_providers()
    
    
    start = time.time()
    y_pred = sess.run(["dense_output"], {"inputs_aux_albedo": xaux, "inputs_main": x_main})[0]
    end = time.time()
    print(end - start)
    # BiGRU+GRU 32 : 4.688 s
    # GRU16 : 1.28s
    # GRU16, softsign: 1.28 still
    #...
    # BiSimple+GRU 32 : 2.3s
    # GRU-BackGru-Gru 16 : 1.56737
    # y_pred[:,0,0] = 1.0
    for i in range(2):
           for j in range(61):
               y_pred[:,j,i] = y_pred[:,j,i] * rsd0
               
    rsd_pred = y_pred[:,:,0]
    rsu_pred = y_pred[:,:,1]
    
    
    plot_flux_and_hr_error(rsu, rsd, rsu_pred, rsd_pred, pres)



# from netCDF4 import Dataset

# rootdir = "../fluxes/"
# fname_out = rootdir+'CAMS_2015_rsud_RADSCHEME_RNN.nc'

# dat_out =  Dataset(fname_out,'a')
# var_rsu = dat_out.variables['rsu']
# var_rsd = dat_out.variables['rsd']

# nsite = dat_out.dimensions['site'].size
# ntime = dat_out.dimensions['time'].size
# var_rsu[:] = rsu_pred.reshape(ntime,nsite,nlay+1)
# var_rsd[:] = rsd_pred.reshape(ntime,nsite,nlay+1)

# dat_out.close()


