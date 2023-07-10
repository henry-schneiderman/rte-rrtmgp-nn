# Same as #6 except can use twice the channels
# Also stripped out everything except direct down path
# And Renormalized inputs to be between 0.0 and 1.0 (changes mostly in RT_net_data.py)

import os
import datetime
import numpy as np
from math import isclose

import tensorflow as tf
from tensorflow.keras import optimizers, Input, Model, initializers
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Dense,TimeDistributed,Layer,RNN,BatchNormalization

from tensorflow.python.framework.ops import disable_eager_execution

# These avoided crashes at some point
#disable_eager_execution()
#tf.compat.v1.experimental.output_all_intermediates(True)

# These are for debugging
#tf.config.run_functions_eagerly(True)
#tf.data.experimental.enable_debug_mode()

from RT_data_hws import load_data_lwp


class OpticalDepth(Layer):
    def __init__(self, n_channels, **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.net = Dense(units=1, 
                         activation='Relu',                              
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.25, seed=None),use_bias=False)
    
    def call(self, input):

        lw, mu = input

        tau = self.net(lw)

        t_direct = tf.math.exp(-tau / (tf.expand_dims(mu,axis=2) + 0.00001))

        return t_direct

    
    def compute_output_shape(self, input_shape):
        # 1 channel
        return [tf.TensorShape([input_shape[0][0],self.n_channels,1])]

    def get_config(self):
        base_config = super(OpticalDepth, self).get_config()
        config = {
            'n_channels': self.n_channels,
        }
        return config.update(base_config)
    @classmethod
    def from_config(cls, config):
        return cls(**config)


    
class DownwardPropagationCell(Layer):
    def __init__(self,n_channels,**kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.state_size = [tf.TensorShape([self.n_channels, 1])]
        self.output_size = tf.TensorShape([self.n_channels, 1])

    def call(self, input_at_i, states_at_i):

        flux_down_above_direct, = states_at_i

        print(f"flux_down_above= {flux_down_above_direct}")

        t_direct = input_at_i

        # Will want this later when incorporate surface interactions
        #absorbed_flux_bottom = flux_down_above_direct * a_bottom_multi_direct + \
        #flux_down_above_diffuse * a_bottom_multi_diffuse

        flux_down_below_direct = flux_down_above_direct * t_direct
        
        output_at_i = flux_down_below_direct
         
        state_at_i_plus_1=[flux_down_below_direct,]

        print ("Downward prop")
        return output_at_i, state_at_i_plus_1

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0],self.n_channels,1])

    def get_config(self):
        base_config = super(DownwardPropagationCell, self).get_config()
        config = {
                'n_channels': self.n_channels,
        }
        return config.update(base_config)
            
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ConsolidateFlux(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, input):
        print("starting: consolidated flux")
        flux_down_above_direct, flux_down_below_direct = input

        # Add layers dimension
        flux_down_above_direct = tf.expand_dims(flux_down_above_direct, axis=1)


        flux_down_direct = tf.concat([flux_down_above_direct,flux_down_below_direct], axis=1)

        # Sum across channels
        flux_down_direct = tf.math.reduce_sum(flux_down_direct, axis=2)
     
        flux_down_direct = tf.squeeze(flux_down_direct,axis=2)

        return flux_down_direct

    def compute_output_shape(self, input_shape):
        return [tf.TensorShape([input_shape[0][0],input_shape[2][1] + 1]),]

class VerificationLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, input):
        flux_down_above_direct, t_direct, flux_down_below_direct = input
        t_output = tf.reduce_sum(tf.reduce_prod(t_direct, axis=1, keepdims=False) * flux_down_above_direct, axis=[1,2])
        flux_output = tf.reduce_sum(flux_down_below_direct[:,-1],axis=[1,2])
        print(f"Verify: t_ouput = {t_output}")
        print(f"Verify: flux_output = {flux_output}")
        print(f"Verify: flux_output  - t_output= {flux_output - t_output}")
        return (t_output - flux_output)
    
    def compute_output_shape(self, input_shape):
        return [tf.TensorShape([input_shape[0][0]])]
    
class HeatingRate(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, input):
        print("starting: heating_rate")
        delta_pressure, absorbed_flux = input
        heating_rate = absorbed_flux_to_heating_rate (absorbed_flux, delta_pressure)
        print("finishing: heating_rate")
        return heating_rate

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class Toa(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, input):
        print("starting: Toa")
        toa = input[0]
        result = toa * 1.0
        print("finishing: Toa")
        return result

class CustomLossWeighted(tf.keras.losses.Loss):
    def __init__(self, weight_profile, **kwargs):
        super().__init__(**kwargs)
        self.weight_profile = weight_profile
    def call(self, y_true, y_pred):
        error = tf.reduce_mean(tf.math.square(self.weight_profile * (y_pred - y_true)))
        return error
    
class CustomLossTOA(tf.keras.losses.Loss):
    def __init__(self, toa, name="weighted_toa", **kwargs):
        super().__init__(name=name, **kwargs)
        self.toa = toa
    def call(self, y_true, y_pred):
        error = tf.reduce_mean(tf.math.square(self.toa * (y_pred - y_true)))
        return error

    def get_config(self):
        config = {
            'toa': self.toa,
        }
        base_config = super().get_config()
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomLossTOA_2(tf.keras.losses.Loss):
    def __init__(self, toa, name="weighted_toa_2", **kwargs):
        super().__init__(name=name, **kwargs)
        self.toa = toa
    def call(self, y_true, y_pred):
        error = tf.reduce_mean(self.toa * (y_pred - y_true))
        return error

class DirectLoss0(tf.keras.losses.Loss):
    def __init__(self, name="direct_loss_0", **kwargs):
        super().__init__(name=name, **kwargs)
    def call(self, y_true, y_pred):
        error = tf.reduce_mean(y_pred - y_true, axis=0)
        error_0 = tf.reduce_sum(error[0:1])
        error_1 = tf.reduce_sum(error[1:5])
        error_2 = tf.reduce_sum(error[5:15])
        error_3 = tf.reduce_sum(error[15:30])
        error_4 = tf.reduce_sum(error[30:])
        return error_0

class DirectLoss1(tf.keras.losses.Loss):
    def __init__(self, name="direct_loss_1", **kwargs):
        super().__init__(name=name, **kwargs)
    def call(self, y_true, y_pred):
        error = tf.reduce_mean(y_pred - y_true, axis=0)
        error_0 = tf.reduce_sum(error[0:1])
        error_1 = tf.reduce_sum(error[1:5])
        error_2 = tf.reduce_sum(error[5:15])
        error_3 = tf.reduce_sum(error[15:30])
        error_4 = tf.reduce_sum(error[30:])
        return error_1

class DirectLoss2(tf.keras.losses.Loss):
    def __init__(self, name="direct_loss_2", **kwargs):
        super().__init__(name=name, **kwargs)
    def call(self, y_true, y_pred):
        error = tf.reduce_mean(y_pred - y_true, axis=0)
        error_0 = tf.reduce_sum(error[0:1])
        error_1 = tf.reduce_sum(error[1:5])
        error_2 = tf.reduce_sum(error[5:15])
        error_3 = tf.reduce_sum(error[15:30])
        error_4 = tf.reduce_sum(error[30:])
        return error_2

class DirectLoss4(tf.keras.losses.Loss):
    def __init__(self, name="direct_loss_4", **kwargs):
        super().__init__(name=name, **kwargs)
    def call(self, y_true, y_pred):
        error = tf.reduce_mean(y_pred - y_true, axis=0)
        error_0 = error[0]
        error_1 = tf.reduce_sum(error[1:5])
        error_2 = tf.reduce_sum(error[5:15])
        error_3 = tf.reduce_sum(error[15:30])
        error_4 = tf.reduce_sum(error[30:])
        return error_4


def train():

    n_hidden_gas = [4, 5]
    n_hidden_layer_coefficients = [4, 5]
    n_layers = 60
    n_composition = 8 # 6 gases + liquid water + ice water
    n_channels = 1 #58
    batch_size  = 2048
    epochs      = 100000
    n_epochs    = 0
    epochs_period = 20
    patience    = 1000 #25

    datadir     = "/home/hws/tmp/"
    filename_training       = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"
    filename_validation   = datadir + "/RADSCHEME_data_g224_CAMS_2014.2.nc"
    filename_testing  = datadir +  "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"
    log_dir = datadir + "/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename_model = datadir + "/Model-"

    # Optical Depth

    lw_input = Input(shape=(n_layers, 1), batch_size=batch_size, name="null_lw_input")

    mu_input = Input(shape=(n_layers, 1), batch_size=batch_size, name="mu_input") 

    optical_depth = TimeDistributed(OpticalDepth(), name="optical_depth")([lw_input, mu_input])

    # Layer coefficients: 
    # direct_transmission, scattered_transmission,
    # scattered_reflection, scattered_absorption


    null_toa_input = Input(shape=(0), batch_size=batch_size, name="null_toa_input")

    #flux_down_above_direct = Dense(units=n_channels,bias_initializer='ones', activation='softmax')(null_toa_input)

    flux_down_above_direct = Dense(units=n_channels,bias_initializer=initializers.RandomUniform(minval=0.1, maxval=1.0), activation='softmax', name='null_toa_dense')(null_toa_input)

    #flux_down_above_direct = tf.keras.layers.Dropout(0.2)(flux_down_above_direct)

    print(f"flux_down_above_direct.shape={flux_down_above_direct.shape}")

    flux_down_above_direct = tf.expand_dims(flux_down_above_direct,2)

    initial_state_down=[flux_down_above_direct,]

    downward_input = optical_depth

    # Downward propagation: t and a
    flux_down_below_direct = RNN(DownwardPropagationCell(n_channels), return_sequences=True, return_state=False, go_backwards=False, time_major=False)(inputs=downward_input, initial_state=initial_state_down)

    flux_inputs = (flux_down_above_direct, flux_down_below_direct)

    flux_down_direct = ConsolidateFlux()(flux_inputs)

    # This is just to force the VerificationLayer to run
    #heating_rate = heating_rate + tf.expand_dims(model_error, axis=1)

    model = Model(inputs=[t_p_input,composition_input,null_lw_input, null_iw_input, null_mu_bar_input, mu_input,null_toa_input], 
    outputs=[flux_down_direct])
    #outputs=[flux_down_direct, flux_down, flux_up,heating_rate, optical_depth])
    #outputs={'flux_down_direct': flux_down_direct, 'flux_down': flux_down, 'flux_up': flux_up, 'heating_rate' : heating_rate})

    training_inputs, training_outputs = load_data_2(filename_training, n_channels)
    validation_inputs, validation_outputs = load_data_2(filename_validation, n_channels)

    #tmp_outputs = model.predict(validation_inputs)

    #print(f"tmp_output / optical depth for gases= {tmp_outputs[4][0]}")

    print(f"flux down direct (after squeeze)= {flux_down_direct.shape}")
    eps = 1.0e-04
    #weight_profile = 1.0 / (eps + tf.math.reduce_mean(flux_down, axis=0, keepdims=True))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.1),
        #loss={flux_down_direct_name: 'mse',flux_down_name:'mse', flux_up_name:'mse', heating_rate_name: 'mse'},
        #loss=['mse', 'mse', 'mse', 'mse'],
        #loss=[CustomLossTOA(toa), CustomLossTOA(toa), CustomLossTOA(toa), CustomLossTOA(toa)],
        loss=[CustomLossTOA(1400.0),],
        #loss={flux_down.name:'mse', flux_up.name : 'mse', heating_rate.name: 'mse'},
        #loss_weights={flux_down_direct_name: 0.1,flux_down_name:0.5, flux_up_name:0.5, heating_rate_name: 0.2},
        #loss_weights= [0.1,0.5,0.5,0.2],
        #loss_weights= [0.1,0.5,0.5,0.8], #TEMP.
        loss_weights= [1.0], #TEMP.3.
        #loss_weights={flux_down.name:0.5, flux_up.name: 0.5, heating_rate.name: 1.0e-4},
        #experimental_run_tf_function=False,
        #metrics={flux_down_direct_name: ['mse'],flux_down_name:['mse'], flux_up_name:['mse'], heating_rate_name: ['mse']},
        #metrics=[['mse'],['mse'],['mse'],['mse']],
        #metrics=[[CustomLossTOA(toa)],[CustomLossTOA(toa)],[CustomLossTOA(toa)],[CustomLossTOA(toa)]],
        metrics=[[CustomLossTOA(1400.0),]],
    #{flux_down.name:'mse', flux_up.name : 'mse', heating_rate.name: 'mse'},
    )
    model.summary()

    print(f"model.metrics_names = {model.metrics_names}")

    
    #output = model(inputs=validation_inputs)

    #print(f"len of output = {len(output)}")

    if False:
        n_epochs = 30
        model.load_weights((filename_model + 'TEMP.3.' + str(n_epochs)))
    writer = tf.summary.create_file_writer(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=False, profile_batch=(4, 10))
    while n_epochs < epochs:
        history = model.fit(x=training_inputs, y=training_outputs,
                epochs = epochs_period, batch_size=batch_size,
                shuffle=True, verbose=1,
                validation_data=(validation_inputs, validation_outputs),
                callbacks = [tensorboard_callback])
                
        #,callbacks = [EarlyStopping(monitor='heating_rate',  patience=patience, verbose=1, \
        #                  mode='min',restore_best_weights=True),])
        
        n_epochs = n_epochs + epochs_period
        print(f"Writing model weights {n_epochs}")
        model.save_weights(filename_model + 'TEMP.3.' + str(n_epochs)) #, save_traces=True)
        
        #del model

        model.load_weights((filename_model + 'TEMP.3.' + str(n_epochs)))
        """ model = tf.keras.models.load_model(filename_model + 'TEMP.' + str(n_epochs),
                                           custom_objects={'OpticalDepth': OpticalDepth,
                                                           'LayerProperties': LayerProperties,
                                                           'UpwardPropagationCell' : UpwardPropagationCell,
                                                           'DownwardPropagationCell' : DownwardPropagationCell,
                                                           'DenseFFN' : DenseFFN,
                                                           }) """

if __name__ == "__main__":
    train()