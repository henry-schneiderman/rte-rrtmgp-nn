# Based on RT_net_hws_9.py but adds functionality for scattering and absorption

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
        self.n_o3 = 13 
        self.n_co2 = 9 
        self.n_u = 13
        self.n_n2o = 3
        self.n_ch4 = 9

        self.net_lw = Dense(units=self.n_channels,
                        activation=tf.keras.activations.relu,                           
                        kernel_initializer=initializers.RandomUniform(minval=0.10, maxval=1.0),use_bias=False)

        self.net_iw = Dense(units=self.n_channels,
                        activation=tf.keras.activations.relu,                             
                        kernel_initializer=initializers.RandomUniform(minval=0.10, maxval=1.0),use_bias=False)

        self.net_h2o = Dense(units=self.n_channels,
                        activation=tf.keras.activations.relu,                            
                        kernel_initializer=initializers.RandomUniform(minval=0.10, maxval=1.0),use_bias=False)
        
        self.net_o3 = Dense(units=self.n_o3,
                        activation=tf.keras.activations.relu,                          
                        kernel_initializer=initializers.RandomUniform(minval=0.10, maxval=1.0),use_bias=False)

        self.net_co2 = Dense(units=self.n_co2,
                        activation=tf.keras.activations.relu,                             
                        kernel_initializer=initializers.RandomUniform(minval=0.10, maxval=1.0),use_bias=False)

        self.net_u = Dense(units=self.n_u,
                        activation=tf.keras.activations.relu,                              
                        kernel_initializer=initializers.RandomUniform(minval=0.10, maxval=1.0),use_bias=False)

                
        self.net_ke_h2o = Dense(units=1,
                        activation=tf.keras.activations.relu,
                        #activation=tf.keras.activations.sigmoid,
                        use_bias=True,                  
                        kernel_initializer='zeros',
                        bias_initializer='ones')

        self.net_n2o = Dense(units=self.n_n2o,
                        activation=tf.keras.activations.relu,                          
                        kernel_initializer=initializers.RandomUniform(minval=0.10, maxval=1.0),use_bias=False)

        self.net_ch4 = Dense(units=self.n_ch4,
                        activation=tf.keras.activations.relu,                          
                        kernel_initializer=initializers.RandomUniform(minval=0.10, maxval=1.0),use_bias=False)
        """ 
        self.net_ke_all = Dense(units=1,
                        activation=tf.keras.activations.relu,
                        #activation=tf.keras.activations.sigmoid,                     
                        #kernel_initializer=initializers.RandomUniform(minval=0.0001, maxval=1.0),use_bias=True)
                        use_bias=True,
                        kernel_initializer='zeros', bias_initializer='ones')
        """
        self.net_ke_o3 = Dense(units=1,
                        activation=tf.keras.activations.relu,
                        #activation=tf.keras.activations.sigmoid,                     
                        #kernel_initializer=initializers.RandomUniform(minval=0.0001, maxval=1.0),use_bias=True)
                        use_bias=True,
                        kernel_initializer='zeros', bias_initializer='ones')

        self.net_ke_u = Dense(units=1,
                        activation=tf.keras.activations.relu,
                        #activation=tf.keras.activations.sigmoid,                     
                        #kernel_initializer=initializers.RandomUniform(minval=0.0001, maxval=1.0),use_bias=True)
                        use_bias=True,
                        kernel_initializer='zeros', bias_initializer='ones')

        self.net_ke_co2 = Dense(units=1,
                        activation=tf.keras.activations.relu,
                        #activation=tf.keras.activations.sigmoid,                     
                        #kernel_initializer=initializers.RandomUniform(minval=0.0001, maxval=1.0),use_bias=True)
                        use_bias=True,
                        kernel_initializer='zeros', bias_initializer='ones')

        self.net_ke_n2o = Dense(units=1,
                        activation=tf.keras.activations.relu,
                        #activation=tf.keras.activations.sigmoid,                     
                        #kernel_initializer=initializers.RandomUniform(minval=0.0001, maxval=1.0),use_bias=True)
                        use_bias=True,
                        kernel_initializer='zeros', bias_initializer='ones')

        self.net_ke_ch4 = Dense(units=1,
                        activation=tf.keras.activations.relu,
                        #activation=tf.keras.activations.sigmoid,                     
                        #kernel_initializer=initializers.RandomUniform(minval=0.0001, maxval=1.0),use_bias=True)
                        use_bias=True,
                        kernel_initializer='zeros', bias_initializer='ones')


    def call(self, input):

        mu, lw, h2o, o3, co2, u, n2o, ch4, t_p = input

        tau_lw = self.net_lw(lw[:,0:1])
        tau_iw = self.net_iw(lw[:,1:2])

        tau_h2o = self.net_h2o(h2o[:,:])

        ke_h2o = self.net_ke_h2o(t_p)
        tau_h2o = tau_h2o * ke_h2o

        ke_o3 = self.net_ke_o3(t_p)

        #self.n_o3 = 13 
        #self.n_co2 = 9 
        #self.n_u = 13
        #self.n_n2o = 3
        #self.n_ch4 = 9

        tau_o3 = self.net_o3(o3[:,:]) * ke_o3
        # amount of padding on each side
        paddings = tf.constant([[0,0],[0,self.n_channels - self.n_o3]])
        tau_o3 = tf.pad(tau_o3, paddings, "CONSTANT")

        ke_co2 = self.net_ke_co2(t_p)
        tau_co2 = self.net_co2(co2[:,:]) * ke_co2
        #overlaps o3 by 3 and no-overlap for 6
        paddings = tf.constant([[0,0],[self.n_o3 - 3, self.n_channels - ((self.n_o3 - 3) + self.n_co2)]])
        tau_co2 = tf.pad(tau_co2, paddings, "CONSTANT")

        ke_u = self.net_ke_u(t_p)
        tau_u = self.net_u(u[:,:]) * ke_u
        # 5 channels
        # overlap with o3 only (2) and o3 + co2 (3)
        paddings_1 = tf.constant([[0,0],[self.n_o3 - 5, self.n_channels - ((self.n_o3 - 5) + 5)]])
        tau_u_1 = tf.pad(tau_u[:,:5], paddings_1, "CONSTANT")

        # Remaining 8 channels: no overlap with o3 or co2
        paddings_2 = tf.constant([[0,0],[(self.n_o3 - 3) + self.n_co2, self.n_channels - ((self.n_o3 - 3) + self.n_co2 + 8)]])
        tau_u_2 = tf.pad(tau_u[:,5:], paddings_2, "CONSTANT")

        ke_n2o = self.net_ke_n2o(t_p)
        tau_n2o = self.net_n2o(n2o[:,:]) * ke_n2o
        #overlaps o3 by 3 and no-overlap for 6
        paddings = tf.constant([[0,0],[self.n_o3 - 3, self.n_channels - ((self.n_o3 - 3) + self.n_n2o)]])
        tau_n2o = tf.pad(tau_n2o, paddings, "CONSTANT")

        ke_ch4 = self.net_ke_ch4(t_p)
        tau_ch4 = self.net_ch4(ch4[:,:]) * ke_ch4
        #overlaps everything by 3
        paddings_a = tf.constant([[0,0],[self.n_o3 - 3, self.n_channels - ((self.n_o3 - 3) + 3)]])
        tau_ch4_1 = tf.pad(tau_ch4[:,:3], paddings_a, "CONSTANT")

        #overlap o2 by 2, no overlap for 2
        paddings_b = tf.constant([[0,0],[self.n_channels - 4, 0]])
        tau_ch4_2 = tf.pad(tau_ch4[:,3:7], paddings_b, "CONSTANT")

        #overlap o3 by 2
        paddings_c = tf.constant([[0,0],[0, self.n_channels - 2]])
        tau_ch4_3 = tf.pad(tau_ch4[:,7:], paddings_c, "CONSTANT")

        tau_u = tau_u_1 + tau_u_2
        tau_ch4 = tau_ch4_1 + tau_ch4_2 + tau_ch4_3
        tau = tau_lw + tau_iw + tau_h2o + tau_o3 + tau_co2 + tau_u + tau_ch4

        t_direct = tf.math.exp(-tau / (mu + 0.0000001))
        t_direct = tf.expand_dims(t_direct, axis=2)

        #print(f"OptficalFlow: shape of tau = {tau.shape}")

        #    return t_direct
        return tau_lw, tau_iw, tau_h2o, tau_o3, tau_co2, tau_u, tau_ch4

    
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
    


#used
def heating_rate_loss(y_true, y_pred, toa_weighting_profile, delta_pressure):
    absorbed_true = toa_weighting_profile * (y_true[:,:-1] - y_true[:,1:])
    absorbed_pred = toa_weighting_profile * (y_pred[:,:-1] - y_pred[:,1:])
    heat_true = absorbed_flux_to_heating_rate(absorbed_true, delta_pressure)
    heat_pred = absorbed_flux_to_heating_rate(absorbed_pred, delta_pressure)
    error = tf.sqrt(tf.reduce_mean(tf.square(heat_true - heat_pred)))
    return error


# Used:
# - weight_profile - determined by average rsd per level [1,*]
# - individual TOAs rather than a single constant [*,1]
def weighted_loss(y_true, y_pred, weight_profile):
    error = tf.reduce_mean(tf.math.square(weight_profile * (y_pred - y_true)))
    return error

def flux_rmse(y_true, y_pred, toa_weighting_profile):
    error = tf.sqrt(weighted_loss(y_true, y_pred, toa_weighting_profile))
    return error
#used
def ukkonen_loss(y_true, y_pred, weight_profile, toa_weighting_profile, delta_pressure):
    flux_loss = weighted_loss(y_true, y_pred, weight_profile)
    hr_loss = heating_rate_loss(y_true, y_pred, toa_weighting_profile, delta_pressure)
    alpha   = 1.0e-4
    return alpha * hr_loss + (1.0 - alpha) * flux_loss

# Used
# A single costant TOA=1400
class OriginalLoss(tf.keras.losses.Loss):
    def __init__(self, toa, name="original", **kwargs):
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
    
def modify_weights_1(model):
    factor_1 = 0.86  # decrease in original positive weights #0.92, 1.1, 0.86
    factor_2 = 0.35  # Initial fraction of possible weight for negative weights #0.1, 0.2, 0.35
    for layer in model.layers:
        if layer.name == 'optical_depth_2':
            layer_weights = layer.get_weights()
            new_weights = []
            for k, weights in enumerate(layer_weights):
                positive_weights = [x for x in np.nditer(weights) if x > 0.0]
                n_positive = len(positive_weights)
                n_negative = weights.size - n_positive
                if n_negative == 0 or k == 6 or k == 7 or k == 10 or k == 11:
                    new_weights.append(weights)
                elif n_positive == 0:
                    new_weights.append(np.full(shape=weights.shape, fill_value=2.0e-02))
                else:
                    new_positive_weight = factor_2 * sum(positive_weights) * (1.0 - factor_1) / n_positive
                    modified_weights = weights * factor_1
                    modified_weights = [x if x > 0.0 else new_positive_weight for x in np.nditer(modified_weights)]
                    np_modified_weights = np.reshape(np.array(modified_weights), weights.shape)
                    new_weights.append(np_modified_weights)
            layer.set_weights(new_weights)
    return model


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
    epochs_period = 50
    patience    = 1000 #25

    datadir     = "/home/hws/tmp/"
    filename_training       = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"
    filename_validation   = datadir + "/RADSCHEME_data_g224_CAMS_2014.2.nc"
    filename_testing  = datadir +  "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"
    log_dir = datadir + "/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename_model = datadir + "/Model-"
    model_name = "Direct_Trans.2."

    # Optical Depth

    mu_input = Input(shape=(n_layers, 1), batch_size=batch_size, name="mu_input") 

    lw_input = Input(shape=(n_layers, 2), batch_size=batch_size, name="lw_input")

    h2o_input = Input(shape=(n_layers, 1), batch_size=batch_size, name="h2o_input") 

    o3_input = Input(shape=(n_layers, 1), batch_size=batch_size, name="o3_input") 

    co2_input = Input(shape=(n_layers, 1), batch_size=batch_size, name="co2_input") 
    u_input = Input(shape=(n_layers, 1), batch_size=batch_size, name="u_input") 

    t_p_input = Input(shape=(n_layers, 2), batch_size=batch_size, name="t_p_input")

    n2o_input = Input(shape=(n_layers, 1), batch_size=batch_size, name="n2o_input") 

    ch4_input = Input(shape=(n_layers, 1), batch_size=batch_size, name="ch4_input") 

    optical_depth = TimeDistributed(OpticalDepth(n_channels), name="optical_depth_2")([mu_input, lw_input, h2o_input, o3_input, co2_input, u_input, n2o_input, ch4_input, t_p_input])

    # Layer coefficients: 
    # direct_transmission, scattered_transmission,
    # scattered_reflection, scattered_absorption


    flux_down_above_direct_input = Input(shape=(1), batch_size=batch_size, name="flux_down_above_direct_input")

    flux_down_above_direct = Dense(units=n_channels,
                                   activation='softmax', 
                                   use_bias=False,
                                   kernel_initializer=initializers.RandomUniform(minval=0.1, maxval=1.0),name='flux_down_above_direct')(flux_down_above_direct_input)

    flux_down_above_direct = tf.expand_dims(flux_down_above_direct, axis=2)

    initial_state_down=[flux_down_above_direct,]

    downward_input = optical_depth

    # Downward propagation: t and a
    flux_down_below_direct = RNN(DownwardPropagationCell(n_channels), return_sequences=True, return_state=False, go_backwards=False, time_major=False)(inputs=downward_input, initial_state=initial_state_down)

    flux_down_direct = ConsolidateFlux()((flux_down_above_direct, flux_down_below_direct))

    toa_input = Input(shape=(1), batch_size=batch_size, name="toa_input")
    target_input = Input(shape=(n_levels), batch_size=batch_size, name="target_input")

    delta_pressure_input = Input(shape=(n_layers), batch_size=batch_size, name="delta_pressure_input")

    # This is just to force the VerificationLayer to run
    #heating_rate = heating_rate + tf.expand_dims(model_error, axis=1)

    model = Model(inputs=[mu_input, lw_input, h2o_input, o3_input, co2_input, u_input, n2o_input, ch4_input, t_p_input, flux_down_above_direct_input, toa_input, target_input, delta_pressure_input], 
    outputs=[flux_down_direct])
    #outputs=[flux_down_direct, flux_down, flux_up,heating_rate, optical_depth])
    #outputs={'flux_down_direct': flux_down_direct, 'flux_down': flux_down, 'flux_up': flux_up, 'heating_rate' : heating_rate})

    training_inputs, training_outputs = load_data_lwp(filename_training, n_channels)
    validation_inputs, validation_outputs = load_data_lwp(filename_validation, n_channels)

    #tmp_outputs = model.predict(validation_inputs)

    #print(f"tmp_output / optical depth for gases= {tmp_outputs[4][0]}")

    #print(f"flux down direct (after squeeze)= {flux_down_direct.shape}")
    eps = 1.0e-04
    #weight_profile = 1.0 / (eps + tf.math.reduce_mean(flux_down, axis=0, keepdims=True))

    weight_profile = 1.0 / tf.reduce_mean(training_outputs[0], axis=0, keepdims=True)

    model.add_metric(heating_rate_loss(target_input, flux_down_direct, toa_input, delta_pressure_input),name="hr")

    model.add_metric(flux_rmse(target_input, flux_down_direct, toa_input),name="flux_rmse")
    
    model.add_loss(ukkonen_loss(target_input, flux_down_direct, weight_profile,toa_input, delta_pressure_input))
    


    model.compile(
        #optimizer=optimizers.Adam(learning_rate=0.1),
        optimizer=optimizers.Adam(),
        #loss={flux_down_direct_name: 'mse',flux_down_name:'mse', flux_up_name:'mse', heating_rate_name: 'mse'},
        #loss=['mse', 'mse', 'mse', 'mse'],
        #loss=[OriginalLoss(toa), OriginalLoss(toa), OriginalLoss(toa), OriginalLoss(toa)],
        #loss=[OriginalLoss(1400.0),],  ****

        #loss={flux_down.name:'mse', flux_up.name : 'mse', heating_rate.name: 'mse'},
        #loss_weights={flux_down_direct_name: 0.1,flux_down_name:0.5, flux_up_name:0.5, heating_rate_name: 0.2},
        #loss_weights= [0.1,0.5,0.5,0.2],
        #loss_weights= [0.1,0.5,0.5,0.8], #TEMP.
        loss_weights= [1.0], #TEMP.4.
        #loss_weights={flux_down.name:0.5, flux_up.name: 0.5, heating_rate.name: 1.0e-4},
        #experimental_run_tf_function=False,
        #metrics={flux_down_direct_name: ['mse'],flux_down_name:['mse'], flux_up_name:['mse'], heating_rate_name: ['mse']},
        #metrics=[['mse'],['mse'],['mse'],['mse']],
        #metrics=[[OriginalLoss(toa)],[OriginalLoss(toa)],[OriginalLoss(toa)],[OriginalLoss(toa)]],
        #metrics=[[AvgWeightLoss(weight_profile)]],  **
        metrics=[[OriginalLoss(1400)]],
    #{flux_down.name:'mse', flux_up.name : 'mse', heating_rate.name: 'mse'},
    )
    model.summary()

    print(f"model.metrics_names = {model.metrics_names}")
    for layer in model.layers:
        print(layer.name, layer)

    
    #output = model(inputs=validation_inputs)

    #print(f"len of output = {len(output)}")

    if False:
        n_epochs = 200
        model.load_weights((filename_model + model_name + str(n_epochs)))
        for layer in model.layers:
            if layer.name == 'flux_down_above_direct':
                print(f'flux_down_above_direct.weights = {layer.weights}')
            if layer.name == 'optical_depth_2':
                print("Optical Depth layers")
                for k, weights in enumerate(layer.weights):
                    print(f'Weights {k}: {weights}')

        model = modify_weights_1(model)
        #writer = tf.summary.create_file_writer(log_dir)
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=False, profile_batch=(4, 10))
    if False:
        n_epochs = 900
        model.load_weights((filename_model + model_name + str(n_epochs)))
        model = modify_weights_1(model)
        #model = modify_weights_1(model)
        #writer = tf.summary.create_file_writer(log_dir)
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=False, profile_batch=(4, 10))
    if True:
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
                    print(f'flux_down_above_direct.weights = {layer.weights}')
                if layer.name == 'optical_depth_2':
                    if False:
                        print(f'optical_depth.weights = {layer.weights[0]}')
                        if n_epochs > epochs_period: #170: #epochs_period:
                            print(f'diff = {tf.convert_to_tensor(last_weights) - tf.convert_to_tensor(layer.weights[0])}')
                        last_weights = tf.identity(layer.weights[0])
                        print("")
                    print("")
                    for k, weights in enumerate(layer.weights):
                        print(f'Weights {k}: {weights}')


            print(f"Writing model weights {n_epochs}")
            model.save_weights(filename_model + model_name + str(n_epochs)) #, save_traces=True)
            #model = modify_weights_1(model)
            #del model

            #model.load_weights((filename_model + 'TEMP.4.' + str(n_epochs)))
            """ model = tf.keras.models.load_model(filename_model + 'TEMP.' + str(n_epochs),
                                            custom_objects={'OpticalDepth': OpticalDepth,
                                                            'LayerProperties': LayerProperties,
                                                            'UpwardPropagationCell' : UpwardPropagationCell,
                                                            'DownwardPropagationCell' : DownwardPropagationCell,
                                                            'DenseFFN' : DenseFFN,
                                                            }) """

if __name__ == "__main__":
    train()