import os

import numpy as np
from math import isclose

import tensorflow as tf
from tensorflow.keras import optimizers, Input, Model, initializers
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Dense,TimeDistributed,Layer,RNN

from tensorflow.python.framework.ops import disable_eager_execution

#disable_eager_execution()
#tf.config.run_functions_eagerly(True)
#tf.data.experimental.enable_debug_mode()
#tf.compat.v1.experimental.output_all_intermediates(True)

from RT_data_hws import absorbed_flux_to_heating_rate, load_data

class DenseFFN(Layer):
    """
    n_hidden[n_layers]: array of the number of nodes per layer
    Last layer has RELU activation insuring non-negative output
    """
    def __init__(self, n_hidden, n_outputs, minval, maxval):
        super().__init__()
        self.hidden = [Dense(units=n, activation='relu',kernel_initializer=initializers.RandomUniform(minval=minval, maxval=maxval), bias_initializer=initializers.RandomNormal()) for n in n_hidden]
        # RELU insures that absorption coefficient is non-negative
        self.out = Dense(units=n_outputs, activation='softplus',kernel_initializer=tf.keras.initializers.glorot_uniform()) 

    def call(self, X):
        for hidden in self.hidden:
            X = hidden(X)
        return self.out(X)
    
class OpticalDepth(Layer):
    def __init__(self, n_hidden, n_channels):
        super().__init__()

        self._n_channels = n_channels

        # h2o, o3, co2, n2o, ch4, uniform
        n_ke = np.array([29,13,9,3,9,13])

        self.ke_gas_net = []

        # Represents a function of temperature and pressure
        # used to build a gas absorption coefficient, ke 

        for n in n_ke:
            self.ke_gas_net.append([DenseFFN(n_hidden,1,minval=20.0,maxval=1000.0) for _ in np.arange(n)])

        self.ke_lw_net = [Dense(units=1,bias_initializer=initializers.RandomUniform(minval=50.0, maxval=10000.0), activation='softplus',) for _ in np.arange(n_channels)]
        self.ke_iw_net = [Dense(units=1, bias_initializer=initializers.RandomUniform(minval=50.0, maxval=10000.0), activation='softplus') for _ in np.arange(n_channels)]

    # Note Ukkonen does not include nitrogen dioxide (no2) in simulation that generated data
    def call(self, input):

        t_p, composition, null_lw, null_iw = input

        # Generate multiple optical depths for each gas

        print(f"shape of composition = {composition.shape}")

        tau_gas = []
        for i, ke_gas_net in enumerate(self.ke_gas_net):
            # Extinction coefficient determined by network
            ke = [net(t_p) for net in ke_gas_net]
            # Tau = ke * mass_path_for_gas
            ke = tf.convert_to_tensor(ke)
            print(f"shape of ke = {ke.shape}")
            tau_gas.append(tf.multiply(ke,tf.reshape(composition[:,i],(1,-1, 1))))

        h2o, o3, co2, n2o, ch4, u = tau_gas

        # Optical depth for each channel
        # using various combinations of gases' optical depths

        #tau_gases = tf.Variable(initial_value=np.zeros((self._n_channels, h2o.shape[1])))

        tau_gases = []

        tau_gases.append(h2o[0] + o3[0] + \
            co2[0] + n2o[0] + ch4[0] + u[0])
        tau_gases.append(h2o[1] + o3[1] + \
            co2[1] + n2o[1] + ch4[1] + u[1])
        tau_gases.append(h2o[2] + o3[2] + \
            co2[2] + n2o[2] + ch4[2] + u[2])

        tau_gases.append(h2o[3] + ch4[3])
        tau_gases.append(h2o[4] + ch4[4])

        tau_gases.append(h2o[5] + co2[3])
        tau_gases.append(h2o[6] + co2[4])

        tau_gases.append(h2o[7] + ch4[5])
        tau_gases.append(h2o[8] + ch4[6])

        tau_gases.append(h2o[9]  + co2[5])
        tau_gases.append(h2o[10] + co2[6])

        tau_gases.append(h2o[11] + ch4[7])
        tau_gases.append(h2o[12] + ch4[8])

        tau_gases.append(h2o[13] + co2[7])
        tau_gases.append(h2o[14] + co2[8])

        tau_gases.append(h2o[15] + u[3])
        tau_gases.append(h2o[16] + u[4])

        tau_gases.append(h2o[17] + o3[3] + u[5])
        tau_gases.append(h2o[18] + o3[4] + u[6])

        tau_gases.append(h2o[19] + o3[5] + u[7])
        tau_gases.append(h2o[20] + o3[6] + u[8])

        tau_gases.append(h2o[21] + o3[7] + u[9])
        tau_gases.append(h2o[22] + o3[8] + u[10])

        tau_gases.append(h2o[23])
        tau_gases.append(h2o[24])

        tau_gases.append(h2o[25] + o3[9])
        tau_gases.append(h2o[26] + o3[10])

        tau_gases.append(h2o[27] + o3[11] + u[11])
        tau_gases.append(h2o[28] + o3[12] + u[12])

        tau_gases = tf.convert_to_tensor(tau_gases)

        # Optical depth for liquid and ice water for each channel
        tau_lw = [net(null_lw) for net in self.ke_lw_net]
        tau_lw = tf.convert_to_tensor(tau_lw)
        tau_lw = tf.multiply(tau_lw,tf.reshape(composition[:,6], (1,-1, 1)))

        tau_iw = [net(null_iw) for net in self.ke_iw_net]
        tau_iw = tf.convert_to_tensor(tau_iw)
        tau_iw = tf.multiply(tau_iw,tf.reshape(composition[:,7], (1,-1, 1)))

        tau_gases = tf.transpose(tau_gases, perm=[1,0,2])
        tau_lw = tf.transpose(tau_lw, perm=[1,0,2])
        tau_iw = tf.transpose(tau_iw, perm=[1,0,2])

        return [tau_gases, tau_lw, tau_iw]

class LayerProperties(Layer):
    def __init__(self, n_hidden, n_channels):
        super().__init__()
        self.extinction_net = [DenseFFN(n_hidden,3,minval=0.0,maxval=1.0) for _ in np.arange(n_channels)]

    def call(self, input):

        tau_gases, tau_lw, tau_iw, mu, mu_bar = input

        print(f"LayerProperties(): shape of tau_gases = {tau_gases.shape}")

        # Iterate over channels

        e_split_direct = [net(tf.concat([tau_gases[:,k], tau_lw[:,k], tau_iw[:,k], mu], axis=1)) for k, net in enumerate(self.extinction_net)]

        e_split_diffuse = [net(tf.concat([tau_gases[:,k], tau_lw[:,k], tau_iw[:,k], mu_bar], axis=1)) for k, net in enumerate(self.extinction_net)]

        e_split_direct = tf.nn.softmax(e_split_direct,axis=-1)
        e_split_diffuse = tf.nn.softmax(e_split_diffuse,axis=-1)

        print(f'Shape of e_split_diffuse = {e_split_diffuse.shape}')
        print(" ")

        # Coefficients of direct transmission of radiation. 
        # Note that diffuse radiation can be directly transmitted

        tau_total = tau_gases + tau_lw + tau_iw

        print(f'Shape of mu = {mu.shape}')
        print(" ")

        print(f'Shape of mu_bar = {mu_bar.shape}')
        print(" ")

        t_direct = tf.math.exp(-tau_total / tf.expand_dims(mu,axis=2))

        print(f'Shape of t_direct = {t_direct.shape}')
        print(" ")

        # To avoid division by zero
        t_diffuse = tf.math.exp(-tau_total / (tf.expand_dims(mu_bar,axis=2) + 0.05))

        e_split_direct = tf.transpose(e_split_direct,perm=[1,0,2])
        e_split_diffuse = tf.transpose(e_split_diffuse,perm=[1,0,2])

        print(f'Shape of e_split_diffuse = {e_split_diffuse.shape}')
        print(" ")

        layer_properties = tf.concat([t_direct, t_diffuse, e_split_direct, e_split_diffuse], axis=2)

        return layer_properties

@tf.function
def propagate_layer_up (t_direct, t_diffuse, e_split_direct, e_split_diffuse, r_bottom_direct, r_bottom_diffuse, a_bottom_direct, a_bottom_diffuse):
    """
    Combines the properties of two atmospheric layers within a column: 
    a shallow "top layer" and a thicker "bottom layer" spanning all the 
    layers beneath the top layer including the surface. Computes the impact of multi-reflection between these layers.

    Naming conventions:
     
    The prefixes -- t, e, r, a -- correspond respectively to absorption,
    extinction, reflection, absorption.

    The suffixes "_direct" and "_diffuse" specify the type of input radiation. 
    Note, however, that an input of direct radiation may produce diffuse output,
    e.g., t_multi_direct (transmission of direct radiation through multi-reflection) 
    
    Input and Output Shape:
        Tensor with shape (n_batches, n_channels)

    Arguments:

        t_direct, t_diffuse - Direct transmission coefficient for 
            the top layer. (Note that diffuse radiation can be directly 
            transmitted)

        e_split_direct, e_split_diffuse - The split of extinguised  
            radiation into diffusely transmitted, reflected,
            and absorbed components. These components sum to 1.0.
            Has additional axis of length=3.
            
        r_bottom_direct, r_bottom_diffuse - The reflection 
            coefficients for bottom layer.

        a_bottom_direct, a_bottom_diffuse - The absorption coefficients
            for the bottom layer. 
            
    Returns:

        t_multi_direct, t_multi_diffuse - The transmission coefficients for 
            radiation that is multi-reflected (as opposed to directly transmitted, 
            e.g., t_direct, t_diffuse)

        r_multi_direct, r_multi_diffuse - The reflection coefficients 
            for the combined top and bottom layers including the surface

        r_bottom_multi_direct, r_bottom_multi_diffuse - The reflection coefficients for
            the bottom layer after accounting for multi-reflection with top layer

        a_top_multi_direct, a_top_multi_diffuse - The absorption coefficients of 
            the top layer after multi-reflection between the layers

        a_bottom_multi_direct, a_bottom_multi_diffuse - The absorption coefficients 
            of the bottom layer after multi-reflection between the layers

    Notes:
        Since the bottom layer includes the surface:
                a_bottom_direct + r_bottom_direct = 1.0
                a_bottom_diffuse + r_bottom_diffuse = 1.0

        Consider two downward fluxes entering top layer: flux_direct, flux_diffuse

            Downward Direct Flux Transmitted = flux_direct * t_direct
            Downward Diffuse Flux Transmitted = flux_direct * t_multi_direct + 
                                            flux_diffuse * (t_diffuse + t_multi_diffuse)

            Upward Flux from Top Layer = flux_direct * r_multi_direct +
                                     flux_diffuse * r_multi_diffuse

            Upward Flux into Top Layer = flux_direct * r_bottom_multi_direct +
                                        flux_diffuse * r_bottom_multi_diffuse

            Both upward fluxes are diffuse since they are from radiation
            that is scattered upwards

        Conservation of energy:
            a_bottom_multi_direct + a_top_multi_direct + r_multi_direct = 1.0
            a_bottom_multi_diffuse + a_top_multi_diffuse + r_multi_diffuse = 1.0

        The absorption at the top layer (after accounting for multi-reflection)
        must equal the combined loss of flux for the downward and upward paths:
         
            a_top_multi_direct = (1 - t_direct - t_multi_direct) + 
                                (r_bottom_multi_direct - r_multi_direct)
            a_top_multi_diffuse = (1 - t_diffuse - t_multi_diffuse) + 
                                (r_bottom_multi_diffuse - r_multi_diffuse)

    """
    # The top layer splits the direct beam into transmitted and extinguished components
    e_direct = 1.0 - t_direct
    
    # The top layer also splits the downward diffuse flux into transmitted and extinguished components
    e_diffuse = 1.0 - t_diffuse

    # The top layer further splits each extinguished component into transmitted, reflected,
    # and absorbed components
    e_t_direct, e_r_direct, e_a_direct = e_split_direct[:,:,0:1], e_split_direct[:,:,1:2],e_split_direct[:,:,2:]
    e_t_diffuse, e_r_diffuse, e_a_diffuse = e_split_diffuse[:,:,0:1], e_split_diffuse[:,:,1:2],e_split_diffuse[:,:,2:]

    # Multi-reflection between the top layer and lower layer resolves 
    # a direct beam into:
    #   r_multi_direct - total effective reflection at the top layer
    #   a_top_multi_direct - absorption at the top layer
    #   a_bottom_multi_direct - absorption for the entire bottom layer

    # The adding-doubling method computes these
    # See p.418-424 of "A First Course in Atmospheric Radiation (2nd edition)"
    # by Grant W. Petty
    #
    # Also see Shonk and Hogan, 2007

    # pre-compute denominator. Add constant to avoid division by zero
    eps = 1.0e-04
    d = 1.0 / (1.0 - e_diffuse * e_r_diffuse * r_bottom_diffuse + eps)

    t_multi_direct = t_direct * r_bottom_direct * e_diffuse * e_r_diffuse * d + \
        e_direct * e_t_direct * d
    
    a_bottom_multi_direct = t_direct * a_bottom_direct + t_multi_direct * a_bottom_diffuse

    r_bottom_multi_direct = t_direct * r_bottom_direct * d + e_direct * e_t_direct * r_bottom_diffuse * d

    a_top_multi_direct = e_direct * e_a_direct + r_bottom_multi_direct * e_diffuse*e_a_diffuse

    r_multi_direct = e_direct * e_r_direct + r_bottom_multi_direct * (t_diffuse + e_diffuse*e_t_diffuse)

    # These should sum to 1.0
    total_direct = a_bottom_multi_direct + a_top_multi_direct + r_multi_direct
    #assert isclose(total_direct, 1.0, abs_tol=1e-5)
    # Loss of flux should equal absorption
    diff_flux = 1.0 - t_direct - t_multi_direct + r_bottom_multi_direct - r_multi_direct 
    #assert isclose(diff_flux, a_top_multi_direct, abs_tol=1e-5)

    # Multi-reflection for diffuse flux

    t_multi_diffuse = \
        t_diffuse * r_bottom_diffuse * e_diffuse * e_r_diffuse * d + \
        e_diffuse * e_t_diffuse * d
    
    a_bottom_multi_diffuse = t_diffuse * a_bottom_diffuse + t_multi_diffuse * a_bottom_diffuse

    r_bottom_multi_diffuse = t_diffuse * r_bottom_diffuse * d + e_diffuse * e_t_diffuse * r_bottom_diffuse * d
    
    a_top_multi_diffuse = e_diffuse * e_a_diffuse + r_bottom_multi_diffuse * e_diffuse*e_a_diffuse

    r_multi_diffuse = e_diffuse * e_r_diffuse + r_bottom_multi_diffuse * (t_diffuse + e_diffuse*e_t_diffuse)

    total_diffuse = a_bottom_multi_diffuse + a_top_multi_diffuse + r_multi_diffuse
    #assert isclose(total_diffuse, 1.0, abs_tol=1e-5)
    diff_flux = 1.0 - t_diffuse - t_multi_diffuse + r_bottom_multi_diffuse - r_multi_diffuse
    #assert isclose(diff_flux, a_top_multi_diffuse, abs_tol=1e-5)

    return t_multi_direct, t_multi_diffuse, \
            r_multi_direct, r_multi_diffuse, \
            r_bottom_multi_direct, r_bottom_multi_diffuse, \
            a_top_multi_direct, a_top_multi_diffuse, \
            a_bottom_multi_direct, a_bottom_multi_diffuse

class UpwardPropagationCell(Layer):
    def __init__(self, n_channels, **kwargs):
        super().__init__(**kwargs)
        #self.state_size = ((n_channels, 1), (n_channels,1), (n_channels, 1), (n_channels, 1))
        #self.state_size = (n_channels * 4)
        self.state_size = [tf.TensorShape([n_channels * 4])]
        self.output_size = tf.TensorShape([n_channels, 8])
        self._n_channels = n_channels

    def call(self, input_at_i, states_at_i):
        print("***")
        t_direct, t_diffuse, e_split_direct, e_split_diffuse = input_at_i[:,:,0:1], input_at_i[:,:,1:2], input_at_i[:,:,2:5],input_at_i[:,:,5:]

        print(f"Enter upward RNN, state.len = {len(states_at_i)} and state[0].shape = {states_at_i[0].shape}")
        print(f"t_direct  = {tf.get_static_value(t_direct)}")

        reshaped_state = tf.reshape(states_at_i[0], (-1,self._n_channels,4))

        r_bottom_direct, r_bottom_diffuse, a_bottom_direct, a_bottom_diffuse = reshaped_state[:,:,0:1], reshaped_state[:,:,1:2], reshaped_state[:,:,1:2], reshaped_state[:,:,2:3]
        
        print(f"r_bottom_direct shape = {r_bottom_direct.shape}")

        tmp = propagate_layer_up (t_direct, t_diffuse, e_split_direct, e_split_diffuse, r_bottom_direct, r_bottom_diffuse, a_bottom_direct, a_bottom_diffuse)

        t_multi_direct, t_multi_diffuse, \
            r_multi_direct, r_multi_diffuse, \
            r_bottom_multi_direct, r_bottom_multi_diffuse, \
            a_top_multi_direct, a_top_multi_diffuse, \
            a_bottom_multi_direct, a_bottom_multi_diffuse= tmp

        output_at_i = tf.concat([t_multi_direct, t_multi_diffuse, 
                                 r_bottom_multi_direct, r_bottom_multi_diffuse,
        a_top_multi_direct, a_top_multi_diffuse,  
        a_bottom_multi_direct, a_bottom_multi_diffuse], axis=2)

        print(f"Upward Prop, r_multi_direct.shape = {r_multi_direct.shape}")
        
        state_at_i_plus_1 = tf.concat([r_multi_direct, r_multi_diffuse, a_top_multi_direct, a_top_multi_diffuse], axis=2)



        print("*")
        state_at_i_plus_1 = tf.reshape(state_at_i_plus_1,(-1,self._n_channels * 4))
        print("**")
        print(" ")
        return output_at_i, [state_at_i_plus_1]


class DownwardPropagationCell(Layer):
    def __init__(self,n_channels):
        super().__init__()
        self.state_size = (n_channels * 2)
        self.output_size = (n_channels, 4)
        self._n_channels = n_channels

    def call(self, input_at_i, states_at_i):

        s = tf.reshape(states_at_i[0],(-1,self._n_channels,2))
        flux_down_above_direct, flux_down_above_diffuse = s[:,:,0:1], s[:,:,1:2]

        i = input_at_i

        t_direct, t_diffuse, \
        t_multi_direct, t_multi_diffuse, \
        r_bottom_multi_direct, r_bottom_multi_diffuse, \
        a_top_multi_direct, a_top_multi_diffuse  = i[:,:,0:1], i[:,:,1:2],i[:,:,2:3], i[:,:,3:4],i[:,:,4:5], i[:,:,5:6],i[:,:,6:7], i[:,:,7:8]

        absorbed_flux_top = flux_down_above_direct * a_top_multi_direct + \
                        flux_down_above_diffuse * a_top_multi_diffuse

        # Will want this later when incorporate surface interactions
        #absorbed_flux_bottom = flux_down_above_direct * a_bottom_multi_direct + \
        #flux_down_above_diffuse * a_bottom_multi_diffuse

        flux_down_below_direct = flux_down_above_direct * t_direct
        flux_down_below_diffuse = flux_down_above_direct * t_multi_direct + \
                                flux_down_above_diffuse * (t_diffuse + t_multi_diffuse)
        flux_up_below_diffuse = flux_down_above_direct * r_bottom_multi_direct + \
                            flux_down_above_diffuse * r_bottom_multi_diffuse
        
        output_at_i = tf.concat([flux_down_below_direct, flux_down_below_diffuse, \
            flux_up_below_diffuse, absorbed_flux_top], axis=2) #, #absorbed_flux_bottom
         
        #state_at_i_plus_1 = flux_down_below_direct, flux_down_below_diffuse
        state_at_i_plus_1=tf.concat([flux_down_above_direct, flux_down_above_diffuse], axis=2)
        state_at_i_plus_1=tf.reshape(state_at_i_plus_1,(-1,self._n_channels*2))

        return output_at_i, state_at_i_plus_1

class CustomLossWeighted(tf.keras.losses.Loss):
    def __init__(self, weight_profile):
        super().__init__()
        self.weight_profile = weight_profile
    def call(self, y_true, y_pred):
        error = tf.reduce_mean(tf.math.square(self.weight_profile * (y_pred - y_true)))
        return(error)
    
class CustomLossTOA(tf.keras.losses.Loss):
    def __init__(self, toa):
        super().__init__()
        self.toa = toa
    def call(self, y_true, y_pred):
        error = tf.reduce_mean(tf.math.square(self.toa * (y_pred - y_true)))
        return(error)

def train():
    n_hidden_gas = [4, 5]
    n_hidden_layer_coefficients = [4, 5]
    n_layers = 60
    n_composition = 8 # 6 gases + liquid water + ice water
    n_channels = 29
    batch_size  = 2048
    epochs      = 100000
    n_epochs    = 0
    epochs_period = 10
    patience    = 1000 #25

    datadir     = "/home/hws/tmp/"
    filename_training       = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"
    filename_validation   = datadir + "/RADSCHEME_data_g224_CAMS_2014.2.nc"
    filename_testing  = datadir +  "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"

    filename_model = datadir + "/Model-"

    # Optical Depth

    t_p_input = Input(shape=(n_layers,2),
                       batch_size=batch_size, name="t_p_input") 
    
    composition_input = Input(shape=(n_layers,n_composition),
                               batch_size=batch_size, name="composition_input")

    null_lw_input = Input(shape=(n_layers, 0), batch_size=batch_size, name="null_lw_input")

    null_iw_input = Input(shape=(n_layers, 0), batch_size=batch_size, name="null_iw_input")

    optical_depth = TimeDistributed(OpticalDepth(n_hidden_gas, n_channels), name="optical_depth")([t_p_input, composition_input, null_lw_input, null_iw_input])

    # Layer coefficients: 
    # direct_transmission, scattered_transmission,
    # scattered_reflection, scattered_absorption

    # This input is always a constant equal to one
    null_mu_bar_input = Input(shape=(0), batch_size=batch_size, name="null_mu_bar_input") 
    mu_bar = Dense(units=1,bias_initializer=tf.constant_initializer(0.5),activation="sigmoid",name="mu_bar")(null_mu_bar_input)

    mu_bar = tf.repeat(tf.expand_dims(mu_bar,axis=1),repeats=n_layers,axis=1)

    mu_input = Input(shape=(n_layers, 1), batch_size=batch_size, name="mu_input") 

    layer_properties = TimeDistributed(LayerProperties(n_hidden_layer_coefficients, n_channels), name="layer_properties")([*optical_depth, mu_input, mu_bar])

    # Upward propagation: a and r 
    # (Working upwards layer by layer computing
    # absorption and reflection for the entire atmosphere
    # below the current layer)

    # absorption and reflection (albedo) of the surface
    surface_input = Input(shape=(n_channels * 4), batch_size=batch_size, name="surface_input")

    upward_output, upward_state = RNN(UpwardPropagationCell(n_channels), return_sequences=True, return_state=True, go_backwards=True, time_major=False)(inputs=layer_properties, initial_state=surface_input)

    print ("****")
    print(f"upward_state shape={tf.shape(upward_state)}")
    print(f"upward_state shape={tf.shape(upward_state[0])}")
    print(" ")
    upward_state = tf.reshape(upward_state[0],(-1,n_channels,4))
    r_multi_direct = upward_state[:,:,0:1]
    # Downward propagation:
    # Determine flux absorbed at each level
    # Determine downward and upward flux at all levels

    null_toa_input = Input(shape=(0), batch_size=batch_size, name="null_toa_input")

    flux_down_above_direct = Dense(units=n_channels,bias_initializer='random_normal', activation='softmax')(null_toa_input)

    print(f"flux_down_above_direct.shape={flux_down_above_direct.shape}")

    flux_down_above_direct = tf.expand_dims(flux_down_above_direct,2)

    # Set to zeros
    flux_down_above_diffuse = Input(shape=(n_channels, 1), batch_size=batch_size, name="flux_down_above_diffuse")

    flux_up_above_diffuse = tf.multiply(flux_down_above_direct,r_multi_direct)

    initial_state_down=tf.concat([flux_down_above_direct, flux_down_above_diffuse], axis=1)
    print(f"initial_state_down.shape={initial_state_down.shape}")
    print(" ")
    initial_state_down=tf.reshape(initial_state_down,(-1,n_channels*2))

    # Downward propagation: t and a
    downward_output = RNN(DownwardPropagationCell(n_channels), return_sequences=True, return_state=False, go_backwards=True, time_major=False)(inputs=upward_output, initial_state=initial_state_down)

    print(f"downward output shape = {downward_output.shape}")

    flux_down_below_direct, flux_down_below_diffuse, \
            flux_up_below_diffuse, absorbed_flux_top = downward_output[:,:,:,0:1],downward_output[:,:,:,1:2],downward_output[:,:,:,2:3],downward_output[:,:,:,3:4]

    print(f"flux_down_above_direct={flux_down_above_direct.shape}")

    flux_down_above_direct = tf.expand_dims(flux_down_above_direct, axis=1)

    flux_down_direct = tf.concat([flux_down_above_direct,flux_down_below_direct], axis=1)

    flux_down_above_diffuse_tmp = tf.expand_dims(flux_down_above_diffuse, axis=1)

    flux_down_diffuse = tf.concat([flux_down_above_diffuse_tmp,flux_down_below_diffuse], axis=1)

    flux_up_above_diffuse = tf.expand_dims(flux_up_above_diffuse, axis=1)

    flux_up_diffuse = tf.concat([flux_up_above_diffuse, flux_up_below_diffuse], axis=1)

    # Sum across channels
    print(f"flux down direct = {flux_down_direct.shape}")

    flux_down_direct = tf.math.reduce_sum(flux_down_direct, axis=2)
    flux_down_diffuse = tf.math.reduce_sum(flux_down_diffuse, axis=2)

    # All upwelling flux is diffuse
    flux_up = tf.math.reduce_sum(flux_up_diffuse, axis=2)
    flux_up = tf.squeeze(flux_up,axis=2)

    flux_down = flux_down_direct + flux_down_diffuse
    flux_down = tf.squeeze(flux_down,axis=2)
    flux_down_direct = tf.squeeze(flux_down_direct,axis=2)



    print(f"absorbed_flux_top = {absorbed_flux_top.shape}")

    absorbed_flux = tf.math.reduce_sum(absorbed_flux_top, axis=2)
    absorbed_flux = tf.squeeze(absorbed_flux,axis=2)

    # Inputs for metrics and loss
    delta_pressure_input = Input(shape=(n_layers), batch_size=batch_size, name="delta_pressure_input")

    toa_input = Input(shape=(1), batch_size=batch_size, name="toa_input")

    heating_rate = absorbed_flux_to_heating_rate (absorbed_flux, delta_pressure_input)

    print(f"heating rate = {heating_rate.shape}")
    model = Model(inputs=[t_p_input,composition_input,null_lw_input, null_iw_input, null_mu_bar_input, mu_input,surface_input, null_toa_input, toa_input, flux_down_above_diffuse, delta_pressure_input], 
    outputs=[flux_down_direct, flux_down, flux_up, heating_rate])
    #outputs=[flux_down_direct, flux_down, flux_up,heating_rate, optical_depth])
    #outputs={'flux_down_direct': flux_down_direct, 'flux_down': flux_down, 'flux_up': flux_up, 'heating_rate' : heating_rate})

    training_inputs, training_outputs = load_data(filename_training, n_channels)
    validation_inputs, validation_outputs = load_data(filename_validation, n_channels)

    #tmp_outputs = model.predict(validation_inputs)

    #print(f"tmp_output / optical depth for gases= {tmp_outputs[4][0]}")

    print(f"flux down direct (after squeeze)= {flux_down_direct.shape}")
    eps = 1.0e-04
    weight_profile = 1.0 / (eps + tf.math.reduce_mean(flux_down, axis=0, keepdims=True))

    prefix = 'tf_op_layer_'
    tmp = flux_down_direct.name
    flux_down_direct_name = prefix + tmp[:tmp.find(':')]
    tmp = flux_down.name
    flux_down_name = prefix + tmp[:tmp.find(':')]
    tmp = flux_up.name
    flux_up_name = prefix + tmp[:tmp.find(':')]
    tmp = heating_rate.name
    heating_rate_name = prefix + tmp[:tmp.find(':')]
    print(f"flux_down_direct.name = {flux_down.name} {flux_down_direct_name}")
    print(f"flux_down.name = {flux_down.name} {flux_down_name}")
    print(f"flux_up.name = {flux_up.name} {flux_up_name}")
    print(f"heating_rate.name = {heating_rate.name} {heating_rate_name}")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.01),
        #loss={flux_down_direct_name: 'mse',flux_down_name:'mse', flux_up_name:'mse', heating_rate_name: 'mse'},
        #loss=['mse', 'mse', 'mse', 'mse'],
        loss=[CustomLossTOA(toa_input), CustomLossTOA(toa_input), CustomLossTOA(toa_input), CustomLossTOA(toa_input)],
        #loss={flux_down.name:'mse', flux_up.name : 'mse', heating_rate.name: 'mse'},
        #loss_weights={flux_down_direct_name: 0.1,flux_down_name:0.5, flux_up_name:0.5, heating_rate_name: 0.2},
        loss_weights= [0.0,0.5,0.5,0.2],
        #loss_weights={flux_down.name:0.5, flux_up.name: 0.5, heating_rate.name: 1.0e-4},
        experimental_run_tf_function=False,
        #metrics={flux_down_direct_name: ['mse'],flux_down_name:['mse'], flux_up_name:['mse'], heating_rate_name: ['mse']},
        #metrics=[['mse'],['mse'],['mse'],['mse',CustomLossTOA(toa_input)]],
        metrics=[[CustomLossTOA(toa_input)],[CustomLossTOA(toa_input)],[CustomLossTOA(toa_input)],[CustomLossTOA(toa_input)]],
    #{flux_down.name:'mse', flux_up.name : 'mse', heating_rate.name: 'mse'},
    )
    model.summary()



    while n_epochs < epochs:
        history = model.fit(x=training_inputs, y=training_outputs,
                epochs = epochs_period, batch_size=batch_size,
                shuffle=True, verbose=1,
                validation_data=(validation_inputs, validation_outputs))
                
        #,callbacks = [EarlyStopping(monitor='heating_rate',  patience=patience, verbose=1, \
        #                  mode='min',restore_best_weights=True),])
        
        n_epochs = n_epochs + epochs_period
        print(f"Writing model {n_epochs}")
        model.save(filename_model + 'TEMP.' + str(n_epochs))
        
        del model
        model = tf.keras.models.load_model(filename_model + 'TEMP.' + str(n_epochs))
    
if __name__ == "__main__":
    train()