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

from RT_data_hws import load_data_lwp, absorbed_flux_to_heating_rate


class OpticalDepth(Layer):
    def __init__(self, n_channels, **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.n_o3 = 5 #13 #5 #13
        if True:
            self.net = Dense(units=self.n_channels,
                            activation=tf.keras.activations.relu,
                            #activation='gelu',                               
                            kernel_initializer=initializers.RandomUniform(minval=0.10, maxval=1.0),use_bias=False)

            self.net_h2o = Dense(units=self.n_channels,
                            activation=tf.keras.activations.relu,
                            #activation='gelu',                               
                            kernel_initializer=initializers.RandomUniform(minval=0.10, maxval=1.0),use_bias=False)
            """    
            self.net_o3 = Dense(units=self.n_o3,
                            activation=tf.keras.activations.relu,
                            #activation='gelu',                               
                            kernel_initializer=initializers.RandomUniform(minval=0.10, maxval=1.0),use_bias=False)
         
            self.tp1 = Dense(units=4, 
                    activation=tf.keras.layers.Activation('relu'),                              
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None),use_bias=False)
            self.bn_tp1 = BatchNormalization()
            self.tp2 = Dense(units=self.n_channels, 
                    activation=tf.keras.layers.Activation('relu'),                              
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.4, seed=None),use_bias=False)
            self.bn_tp2 = BatchNormalization()
            self.tp3 = Dense(units=self.n_channels, 
                    activation=tf.keras.layers.Activation('relu'),                              
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.4, seed=None),use_bias=False) """
            

            self.tp1_h2o = Dense(units=4, 
                    activation=tf.keras.layers.Activation('elu'),                              
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None),use_bias=False)
            self.bn_tp1_h2o = BatchNormalization()
            self.tp2_h2o = Dense(units=7, #self.n_channels, 
                    activation=tf.keras.layers.Activation('elu'),                              
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.4, seed=None),use_bias=False)
            self.bn_tp2_h2o = BatchNormalization()
            self.tp3_h2o = Dense(units=7, #self.n_channels, 
                    activation=tf.keras.layers.Activation('elu'),                              
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.4, seed=None),use_bias=False)
            self.bn_tp3_h2o = BatchNormalization()
            self.tp4_h2o = Dense(units=1, #self.n_channels, 
                    activation=tf.keras.layers.Activation('relu'),                              
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.4, seed=None),use_bias=False)            
            """
            self.tp1_o3 = Dense(units=4, 
                    activation=tf.keras.layers.Activation('relu'),                              
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None),use_bias=False)
            self.bn_tp1_o3 = BatchNormalization()
            self.tp2_o3 = Dense(units=5, #self.n_o3, 
                    activation=tf.keras.layers.Activation('relu'),                              
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.4, seed=None),use_bias=False)
            self.bn_tp2_o3 = BatchNormalization()
            self.tp3_o3 = Dense(units=5, #self.n_o3, 
                    activation=tf.keras.layers.Activation('relu'),                              
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.4, seed=None),use_bias=False)
            self.bn_tp3_o3 = BatchNormalization()
            self.tp4_o3 = Dense(units=1, #self.n_o3, 
                    activation=tf.keras.layers.Activation('relu'),                              
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.4, seed=None),use_bias=False)
            """
        if False:

            self.net_lw = [Dense(units=1, 
                        activation='relu',                              
                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.25, seed=None),use_bias=False) for i in np.arange(self.n_channels)]

            self.net_iw = [Dense(units=1, 
                        activation='relu',                              
                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.25, seed=None),use_bias=False) for i in np.arange(self.n_channels)]
        if False:
            self.net = [Dense(units=1, 
                            activation='relu',                              
                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.25, seed=None),use_bias=False) for i in np.arange(self.n_channels)]
        if False:
            self.net = Dense(units=4, 
                    activation=tf.keras.layers.Activation('relu'),                              
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.2, seed=None),use_bias=False)

            self.net2 = Dense(units=5, 
                    activation=tf.keras.layers.Activation('relu'),                              
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.2, seed=None),use_bias=False)
            self.bn1 = BatchNormalization()
            self.bn2 = BatchNormalization()
            self.net3 = Dense(units=8, 
                    activation=tf.keras.layers.Activation('relu'),                              
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.2, seed=None),use_bias=False)
            self.net4 = Dense(units=5, 
                    activation=tf.keras.layers.Activation('relu'),                              
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.2, seed=None),use_bias=False)
            self.bn3 = BatchNormalization()
            self.net5 = Dense(units=self.n_channels, 
                    activation=tf.keras.layers.Activation('relu'),                              
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.2, seed=None),use_bias=False)
            self.bn4 = BatchNormalization()
    
            self.tp1 = Dense(units=4, 
                    activation=tf.keras.layers.Activation('relu'),                              
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None),use_bias=False)
            self.bn_tp1 = BatchNormalization()
            self.tp2 = Dense(units=4, 
                    activation=tf.keras.layers.Activation('relu'),                              
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.4, seed=None),use_bias=False)
            self.bn_tp2 = BatchNormalization()
            self.tp3 = Dense(units=self.n_channels, 
                    activation=tf.keras.layers.Activation('relu'),                              
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.4, seed=None),use_bias=False)

            self.v2_1 = Dense(units=6, 
                    activation=tf.keras.layers.Activation('relu'),                              
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None),use_bias=False)
            self.bn_v2_1 = BatchNormalization()

            self.v2_2 = Dense(units=6, 
                    activation=tf.keras.layers.Activation('relu'),                              
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None),use_bias=False)
            
            self.bn_v2_2 = BatchNormalization()
            
            self.v2_3 = Dense(units=6, 
                    activation=tf.keras.layers.Activation('relu'),                              
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None),use_bias=False)
            self.bn_v2_3 = BatchNormalization()

            self.v2_4 = Dense(units=self.n_channels, 
                    activation=tf.keras.layers.Activation('sigmoid'),                              
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.4, seed=None),use_bias=False)
    def call(self, input):

        mu, lw, h2o, o3, t_p = input

        if False:
            tau = [net(lw) for net in self.net]
            tau = tf.convert_to_tensor(tau)
            tau = tf.transpose(tau, perm=[1,0])
            mu_tmp = tf.expand_dims(mu,axis=2)

            print(f"shape of mu_tmp = {mu_tmp.shape}")

            t_direct = tf.math.exp(-tau / (mu_tmp + 0.0000001))
            #t_direct = tf.math.exp(-tau)

        if False:
            tau_lw = [net(lw[:,0:1]) for net in self.net_lw]
            tau_iw = [net(lw[:,1:2]) for net in self.net_iw]

            tau_lw = tf.convert_to_tensor(tau_lw)
            tau_iw = tf.convert_to_tensor(tau_iw)

            tau = tau_lw + tau_iw
            print(f"Optical Depth: shape of tau = {tau.shape}")

            tau = tf.transpose(tau, perm=[1,0,2])

            mu_tmp = tf.expand_dims(mu,axis=2)
            t_direct = tf.math.exp(-tau / (mu_tmp + 0.0000001))


            print(f"shape of tau = {tau.shape}")

        if True:
            #tau = self.net3(self.bn2(self.net2(self.bn1(self.net(lw)))))
            tau = self.net(lw[:,:])
            tau_h2o = self.net_h2o(h2o[:,:])

            #ke = self.tp3(self.bn_tp2(self.tp2(self.bn_tp1(self.tp1(t_p)))))

            #ke = self.tp2(self.bn_tp1(self.tp1(t_p)))
            #tau = tau * ke
            ke_h2o = self.tp4_h2o(self.bn_tp3_h2o(self.tp3_h2o(self.bn_tp2_h2o(self.tp2_h2o(self.bn_tp1_h2o(self.tp1_h2o(t_p)))))))
            tau_h2o = tau_h2o * ke_h2o

            """
            tau_o3 = self.net_o3(o3[:,:])
            ke_o3 = self.tp4_o3(self.bn_tp3_o3(self.tp3_o3(self.bn_tp2_o3(self.tp2_o3(self.bn_tp1_o3(self.tp1_o3(t_p)))))))
            tau_o3 = tau_o3 * ke_o3
            paddings = tf.constant([[0,0],[0,self.n_channels - self.n_o3]])
            tau_o3 = tf.pad(tau_o3, paddings, "CONSTANT")
            """


            mu_tmp = tf.expand_dims(mu,axis=2)

            print(f"shape of mu_tmp = {mu_tmp.shape}")
            tau = tau + tau_h2o #+ tau_o3
            #tau[:,:self.n_o3] = tau[:,:self.n_o3] + tau_o3
            tau = tf.expand_dims(tau, axis=2)
            t_direct = tf.math.exp(-tau / (mu_tmp + 0.0000001))

        if False:
            #tau = self.net(lw)
            tau = self.net5(self.bn4(self.net4(self.bn3(self.net3(self.bn2(self.net2(self.bn1(self.net(lw)))))))))

            ke = self.tp3(self.bn_tp2(self.tp2(self.bn_tp1(self.tp1(t_p)))))
            tau = tau * ke

            mu_tmp = tf.expand_dims(mu,axis=2)

            print(f"shape of mu_tmp = {mu_tmp.shape}")
            tau = tf.expand_dims(tau, axis=2)
            t_direct = tf.math.exp(-tau / (mu_tmp + 0.0000001))

        if False:
            print(f"OptficalFlow: shape of mu = {mu.shape}")
            #mu_tmp = tf.expand_dims(mu,axis=2)
            print(f"OptficalFlow: shape of lw = {lw.shape}")
            lw = lw / (mu + 0.0000001)
            print(f"OptficalFlow: shape of lw = {lw.shape}")
            print(f"OptficalFlow: shape of t_p = {t_p.shape}")
            vars = tf.concat([lw, t_p], axis=1)

            t_direct = self.v2_4(self.bn_v2_3(self.v2_3(self.bn_v2_2(self.v2_2(self.bn_v2_1(self.v2_1(vars)))))))
            t_direct = tf.expand_dims(t_direct,axis=2)

        #print(f"OptficalFlow: shape of tau = {tau.shape}")

        return t_direct

    
    def compute_output_shape(self, input_shape):
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

        print(f"RNN: flux_down_above.shape= {flux_down_above_direct.shape}")

        t_direct = input_at_i

        print(f"RNN: t_direct.shape= {t_direct.shape}")

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

        print(f'Consolidate flux_down.shape = {flux_down_direct.shape}')

        # Sum across channels

        flux_down_direct = tf.math.reduce_sum(flux_down_direct, axis=2)
        flux_down_direct = tf.squeeze(flux_down_direct,axis=2)      
        flux_down_direct = flux_down_direct[:,:]


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

#used
def heating_rate_error(y_true, y_pred, toa_weighting_profile, delta_pressure):
    absorbed_true = toa_weighting_profile * (y_true[:,:-1] - y_true[:,1:])
    absorbed_pred = toa_weighting_profile * (y_pred[:,:-1] - y_pred[:,1:])
    heat_true = absorbed_flux_to_heating_rate(absorbed_true, delta_pressure)
    heat_pred = absorbed_flux_to_heating_rate(absorbed_pred, delta_pressure)
    error = tf.sqrt(tf.reduce_mean(tf.square(heat_true - heat_pred)))
    return error

# Used
def CustomLossWeighted(weight_profile):
    def loss(y_true, y_pred):
        error = tf.reduce_mean(tf.math.square(weight_profile * (y_pred - y_true)))
        return error
    return loss


# Used with individual TOAs rather than a signle constant
def weighted_loss(y_true, y_pred, weight_profile):
    error = tf.reduce_mean(tf.math.square(weight_profile * (y_pred - y_true)))
    return error

# Used
# A single costant TOA
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

# Used
class CustomLoss2(tf.keras.losses.Loss):
    def __init__(self, weight_profile, name="custom_loss_2", **kwargs):
        super().__init__(name=name, **kwargs)
        self.weight_profile = weight_profile
    def call(self, y_true, y_pred):
        error = tf.reduce_mean(tf.math.square(self.weight_profile * (y_pred - y_true)))
        return error

    def get_config(self):
        config = {
            'weight_profile': self.weight_profile,
        }
        base_config = super().get_config()
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)



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
    n_levels = n_layers + 1
    n_composition = 8 # 6 gases + liquid water + ice water
    n_channels = 29 #58
    batch_size  = 2048
    epochs      = 100000
    n_epochs    = 0
    epochs_period = 5
    patience    = 1000 #25

    datadir     = "/home/hws/tmp/"
    filename_training       = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"
    filename_validation   = datadir + "/RADSCHEME_data_g224_CAMS_2014.2.nc"
    filename_testing  = datadir +  "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"
    log_dir = datadir + "/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename_model = datadir + "/Model-"

    # Optical Depth

    mu_input = Input(shape=(n_layers, 1), batch_size=batch_size, name="mu_input") 

    lw_input = Input(shape=(n_layers, 2), batch_size=batch_size, name="lw_input")

    h2o_input = Input(shape=(n_layers, 1), batch_size=batch_size, name="h2o_input") 

    o3_input = Input(shape=(n_layers, 1), batch_size=batch_size, name="o3_input") 

    t_p_input = Input(shape=(n_layers, 2), batch_size=batch_size, name="t_p_input")

    optical_depth = TimeDistributed(OpticalDepth(n_channels), name="optical_depth")([mu_input, lw_input, h2o_input, o3_input, t_p_input])

    # Layer coefficients: 
    # direct_transmission, scattered_transmission,
    # scattered_reflection, scattered_absorption


    flux_down_above_direct_input = Input(shape=(1), batch_size=batch_size, name="flux_down_above_direct_input")

    flux_down_above_direct = Dense(units=n_channels,
                                   activation='softmax', 
                                   use_bias=False,
                                   kernel_initializer=initializers.RandomUniform(minval=0.1, maxval=1.0),name='flux_down_above_direct')(flux_down_above_direct_input)

    flux_down_above_direct = tf.expand_dims(flux_down_above_direct, axis=2)


    constant_flux_down_above_direct_input = Input(shape=(n_channels, 1), batch_size=batch_size, name="constant_flux_down_above_direct_input")

    if True:
        initial_state_down=[flux_down_above_direct,]
    else:
        initial_state_down=[constant_flux_down_above_direct_input,]

    downward_input = optical_depth

    # Downward propagation: t and a
    flux_down_below_direct = RNN(DownwardPropagationCell(n_channels), return_sequences=True, return_state=False, go_backwards=False, time_major=False)(inputs=downward_input, initial_state=initial_state_down)

    if True:
        flux_inputs = (flux_down_above_direct, flux_down_below_direct)
    else:
        # Add layers dimension
        flux_down_above_direct_expanded = tf.expand_dims(flux_down_above_direct, axis=1)
        flux_down_below_direct = flux_down_below_direct * flux_down_above_direct_expanded
        flux_inputs = (flux_down_above_direct, flux_down_below_direct)



    flux_down_direct = ConsolidateFlux()(flux_inputs)

    toa_input = Input(shape=(1), batch_size=batch_size, name="toa_input")
    target_input = Input(shape=(n_levels), batch_size=batch_size, name="target_input")

    delta_pressure_input = Input(shape=(n_layers), batch_size=batch_size, name="delta_pressure_input")

    # This is just to force the VerificationLayer to run
    #heating_rate = heating_rate + tf.expand_dims(model_error, axis=1)

    model = Model(inputs=[mu_input, lw_input, h2o_input, o3_input, t_p_input, flux_down_above_direct_input, constant_flux_down_above_direct_input, toa_input, target_input, delta_pressure_input], 
    outputs=[flux_down_direct])
    #outputs=[flux_down_direct, flux_down, flux_up,heating_rate, optical_depth])
    #outputs={'flux_down_direct': flux_down_direct, 'flux_down': flux_down, 'flux_up': flux_up, 'heating_rate' : heating_rate})

    training_inputs, training_outputs = load_data_lwp(filename_training, n_channels)
    validation_inputs, validation_outputs = load_data_lwp(filename_validation, n_channels)

    #tmp_outputs = model.predict(validation_inputs)

    #print(f"tmp_output / optical depth for gases= {tmp_outputs[4][0]}")

    print(f"flux down direct (after squeeze)= {flux_down_direct.shape}")
    eps = 1.0e-04
    #weight_profile = 1.0 / (eps + tf.math.reduce_mean(flux_down, axis=0, keepdims=True))

    model.add_metric(weighted_loss(target_input, flux_down_direct, toa_input),name="toa_weighted")

    model.add_metric(heating_rate_error(target_input, flux_down_direct, toa_input, delta_pressure_input),name="hr")

    def CustomLossWeighted(weight):
        def loss(y_true, y_pred):
            error = tf.reduce_mean(tf.math.square(weight * (y_pred - y_true)))
            return error
        return loss
    
    weight_profile = 1.0 / tf.reduce_mean(training_outputs[0], axis=0, keepdims=True)
    model.compile(
        #optimizer=optimizers.Adam(learning_rate=0.1),
        optimizer=optimizers.Adam(),
        #loss={flux_down_direct_name: 'mse',flux_down_name:'mse', flux_up_name:'mse', heating_rate_name: 'mse'},
        #loss=['mse', 'mse', 'mse', 'mse'],
        #loss=[CustomLossTOA(toa), CustomLossTOA(toa), CustomLossTOA(toa), CustomLossTOA(toa)],
        loss=[CustomLossTOA(1400.0),],
        #loss=[CustomLossWeighted(toa_input)],
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
        metrics=[[CustomLoss2(weight_profile)]],
    #{flux_down.name:'mse', flux_up.name : 'mse', heating_rate.name: 'mse'},
    )
    model.summary()

    print(f"model.metrics_names = {model.metrics_names}")
    for layer in model.layers:
        print(layer.name, layer)

    
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
                callbacks = []) #[tensorboard_callback])
                
        #,callbacks = [EarlyStopping(monitor='heating_rate',  patience=patience, verbose=1, \
        #                  mode='min',restore_best_weights=True),])
        
        print (" ")

        print (" ")
        n_epochs = n_epochs + epochs_period

        for layer in model.layers:
            if layer.name == 'flux_down_above_direct':
                  print(f'flux_down_above_direct.weights = {tf.keras.activations.softmax(tf.convert_to_tensor(layer.weights))}')
            if layer.name == 'optical_depth':
                  print(f'optical_depth.weights = {layer.weights[0]}')
                  if n_epochs > epochs_period:
                      print(f'diff = {tf.convert_to_tensor(last_weights) - tf.convert_to_tensor(layer.weights[0])}')
                  last_weights = tf.identity(layer.weights[0])


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