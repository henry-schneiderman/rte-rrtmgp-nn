import os

import numpy as np
from math import isclose


import tensorflow as tf
from tensorflow.keras import losses, optimizers, layers, Input, Model, Layer, Sequential
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Dense,TimeDistributed


class NoInputLayer(Layer):
    def __init__(self,num_outputs,minval=0.0,maxval=1.0):
        super().__init__()
        self.w = tf.random.normal((num_outputs),minval=minval,maxval=maxval)
        self.w = tf.Variable(self.w)
    def call(self):
        return(tf.nn.relu(self.w))

class DenseFFN(Layer):
    """
    n_hidden[n_layers]: array of the number of nodes per layer
    Last layer has RELU activation insuring non-negative output
    """
    def __init__(self, n_hidden, n_outputs):
        super().__init__()
        self.hidden = [Dense(units=n, activation='elu',) for n in n_hidden]
        # RELU insures that absorption coefficient is non-negative
        self.out = Dense(units=n_outputs, activation='relu') 

    def call(self, X):
        for hidden in self.hidden:
            X = hidden(X)
        return self.out(X)
    
class OpticalDepth(Layer):
    def __init__(self, n_hidden, n_channels):
        super().__init__()

        self._n_channels = n_channels

        n_gas = {}
        n_gas['h2o']=29
        n_gas['o3'] =13
        n_gas['co2']=9
        n_gas['n2o']=3
        n_gas['ch4']=9
        # 'u' represents all gases that are uniform in concentration in
        # the atmosphere: o2, n2
        # Could probably include co2, too, but it is separated out for
        # now to account for annual differences
        n_gas['uniform']=13  

        self.gas_index = {}
        self.gas_index['h2o']=2
        self.gas_index['o3'] =3
        self.gas_index['co2']=4
        self.gas_index['n2o']=5
        self.gas_index['ch4']=6
        # 'u' represents all gases that are uniform in concentration in
        # the atmosphere: o2, n2
        # Could probably include co2, too, but it is separated out for
        # now to account for annual differences
        self.gas_index['uniform']=7  

        self.ke_gas_net = {}

        # Represents a function of temperature and pressure
        # used to build a gas absorption coefficient, ke 

        for gas,n in n_gas.items():
            self.ke_gas_net[gas] = [DenseFFN(n_hidden,1) for _ in np.arange(n)]

        self.ke_lw_net = [NoInputLayer(1) for _ in np.arange(n_channels)]
        self.ke_iw_net = [NoInputLayer(1) for _ in np.arange(n_channels)]

    # Note Ukkonen does not include nitrogen dioxide (no2) in simulation that generated data
    def call(self, input):
        #t_p, composition, lwp, iwp = input

        t_p = input[0:2]

        # Optical depths for each gas
        tau_gas = {}
        for gas, ke_gas_net in self.ke_gas_net.items():
            # Extinction coefficient determined by network
            ke = [net(t_p) for net in ke_gas_net]
            # Tau = ke * mass_path_for_gas
            gas_index = self.gas_index[gas]
            tau_gas[gas] = tf.multiply(ke,input[gas_index])

        h2o = tau_gas['h2o']
        o3 = tau_gas['o3']
        co2 = tau_gas['co2']
        n2o = tau_gas['n2o']
        ch4 = tau_gas['ch4']
        u = tau_gas['uniform']

        # Optical depth for each channel

        # 0: tau for all gases
        # 1: tau for liquid water
        # 2: tau for ice water
        output = tf.zeros((self._n_channels,3))
 
        # Each channel combines various various extinction coefficients
        # associated with various gases
        output[0,0] = h2o[0] + o3[0] + \
            co2[0] + n2o[0] + ch4[0] + u[0]
        output[1,0] = h2o[1] + o3[1] + \
            co2[1] + n2o[1] + ch4[1] + u[1]
        output[2,0] = h2o[2] + o3[2] + \
            co2[2] + n2o[2] + ch4[2] + u[2]

        output[3,0] = h2o[3] + ch4[3]
        output[4,0] = h2o[4] + ch4[4]

        output[5,0] = h2o[5] + co2[3]
        output[6,0] = h2o[6] + co2[4]

        output[7,0] = h2o[3] + ch4[5]
        output[8,0] = h2o[4] + ch4[6]

        output[9,0]  = h2o[9]  + co2[5]
        output[10,0] = h2o[10] + co2[6]

        output[11,0] = h2o[11] + ch4[7]
        output[12,0] = h2o[12] + ch4[8]

        output[13,0] = h2o[13] + co2[7]
        output[14,0] = h2o[14] + co2[8]

        output[15,0] = h2o[15] + u[3]
        output[16,0] = h2o[16] + u[4]

        output[17,0] = h2o[17] + o3[3] + u[5]
        output[18,0] = h2o[18] + o3[4] + u[6]

        output[19,0] = h2o[19] + o3[5] + u[7]
        output[20,0] = h2o[20] + o3[6] + u[8]

        output[21,0] = h2o[21] + o3[7] + u[9]
        output[22,0] = h2o[22] + o3[8] + u[10]

        output[23,0] = h2o[23]
        output[24,0] = h2o[24]

        output[25,0] = h2o[25] + o3[9]
        output[26,0] = h2o[26] + o3[10]

        output[27,0] = h2o[27] + o3[11] + u[11]
        output[28,0] = h2o[28] + o3[12] + u[12]

        # Optical depth for liquid and ice water for each channel
        tau_lw = [net() for net in self.ke_lw_net]
        output[:,1] = tf.multiply(tau_lw,input[8])
        tau_iw = [net() for net in self.ke_iw_net]
        output[:,2] = tf.multiply(tau_iw,input[9])

        return output

class LayerProperties(Layer):
    def __init__(self, n_hidden, n_channels):
        super().__init__()
        self.extinction_net = [DenseFFN(n_hidden,3) for _ in np.arange(n_channels)]

    def call(self, input):

        # Components of optical depth for each channel
        # tau_gases, tau_lw, tau_iw, mu, mu_bar = input

        # Iterate over channels
        # Uses mu
        e_split_direct = [net(input[k,0],input[k,1],input[k,2], input[k,3]) for k, net in self.extinction_net.items()]
        # Uses mu_bar
        e_split_diffuse = [net(input[k,0],input[k,1],input[k,2], input[k,4]) for k, net in self.extinction_net.items()]

        e_split_direct = tf.nn.softmax(e_split_direct)
        e_split_diffuse = tf.nn.softmax(e_split_diffuse)

        # Direct transmission of radiation. 
        # Note that diffuse radiation can be directly transmitted
        tau_total = input[:,0] + input[:,1] + input[:,2]
        t_direct = tf.math.exp(-tau_total / input[:,3])
        t_diffuse = tf.math.exp(-tau_total / input[:,4])

        return t_direct, t_diffuse, e_split_direct, e_split_diffuse

    def old_call(self, input):

        # Components of optical depth for each channel
        tau_gases, tau_lw, tau_iw, mu, mu_bar = input

        # Total optical depth for each channel
        tau_total = tau_gases + tau_lw + tau_iw

        # Use NN to determine other properties of layer
        input_1 = tau_lw / tau_total
        input_2 = tau_iw / tau_total

        input_3_direct = tau_total / mu
        input_3_diffuse = tau_total / mu_bar

        e_split_direct = [net(input_1[k],input_2[k],input_3_direct[k], mu[k]) for k, net in self.extinction_net.items()]
        e_split_diffuse = [net(input_1[k],input_2[k],input_3_diffuse[k], mu_bar[k]) for k, net in self.extinction_net.items()]

        e_split_direct = tf.nn.softmax(e_split_direct)
        e_split_diffuse = tf.nn.softmax(e_split_diffuse)

        # Direct transmission of radiation. 
        # Note that diffuse radiation can be directly transmitted
        t_direct = tf.exp(-input_3_direct)
        t_diffuse = tf.exp(-input_3_diffuse)

        return t_direct, t_diffuse, e_split_direct, e_split_diffuse



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
    e_direct = 1 - t_direct
    
    # The top layer also splits the downward diffuse flux into transmitted and extinguished components
    e_diffuse = 1 - t_diffuse

    # The top layer further splits each extinguished component into transmitted, reflected,
    # and absorbed components
    e_t_direct, e_r_direct, e_a_direct = tf.transpose(e_split_direct, perm=[2,0,1])
    e_t_diffuse, e_r_diffuse, e_a_diffuse = tf.transpose(e_split_diffuse, perm=[2,0,1])

    # Multi-reflection between the top layer and lower layer resolves 
    # a direct beam into:
    #   r_multi_direct - total effective reflection at the top layer
    #   a_top_multi_direct - absorption at the top layer
    #   a_bottom_multi_direct - absorption for the entire bottom layer

    # The adding-doubling method computes these
    # See p.418-424 of "A First Course in Atmospheric Radiation (2nd edition)"
    # by Grant W. Petty

    # pre-compute denominator
    d = 1.0 / (1.0 - e_diffuse * e_r_diffuse * r_bottom_diffuse)

    t_multi_direct = t_direct * r_bottom_direct * e_diffuse * e_r_diffuse * d + \
        e_direct * e_t_direct * d
    
    a_bottom_multi_direct = t_direct * a_bottom_direct + t_multi_direct * a_bottom_diffuse

    r_bottom_multi_direct = t_direct * r_bottom_direct * d + e_direct * e_t_direct * r_bottom_diffuse * d

    a_top_multi_direct = e_direct * e_a_direct + r_bottom_multi_direct * e_diffuse*e_a_diffuse

    r_multi_direct = e_direct * e_r_direct + r_bottom_multi_direct * (t_diffuse + e_diffuse*e_t_diffuse)

    # These should sum to 1.0
    total_direct = a_bottom_multi_direct + a_top_multi_direct + r_multi_direct
    assert isclose(total_direct, 1.0, abs_tol=1e-5)
    # Loss of flux should equal absorption
    diff_flux = 1 - t_direct - t_multi_direct + r_bottom_multi_direct - r_multi_direct 
    assert isclose(diff_flux, a_top_multi_direct, abs_tol=1e-5)

    # Multi-reflection for diffuse flux

    t_multi_diffuse = \
        t_diffuse * r_bottom_diffuse * e_diffuse * e_r_diffuse * d + \
        e_diffuse * e_t_diffuse * d
    
    a_bottom_multi_diffuse = t_diffuse * a_bottom_diffuse + t_multi_diffuse * a_bottom_diffuse

    r_bottom_multi_diffuse = t_diffuse * r_bottom_diffuse * d + e_diffuse * e_t_diffuse * r_bottom_diffuse * d
    
    a_top_multi_diffuse = e_diffuse * e_a_diffuse + r_bottom_multi_diffuse * e_diffuse*e_a_diffuse

    r_multi_diffuse = e_diffuse * e_r_diffuse + r_bottom_multi_diffuse * (t_diffuse + e_diffuse*e_t_diffuse)

    total_diffuse = a_bottom_multi_diffuse + a_top_multi_diffuse + r_multi_diffuse
    assert isclose(total_diffuse, 1.0, abs_tol=1e-5)
    diff_flux = 1 - t_diffuse - t_multi_diffuse + r_bottom_multi_diffuse - r_multi_diffuse
    assert isclose(diff_flux, a_top_multi_diffuse, abs_tol=1e-5)

    return t_multi_direct, t_multi_diffuse, \
            r_multi_direct, r_multi_diffuse, \
            r_bottom_multi_direct, r_bottom_multi_diffuse, \
            a_top_multi_direct, a_top_multi_diffuse, \
            a_bottom_multi_direct, a_bottom_multi_diffuse

class UpwardPropagationCell(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state_size = ???
        self.output_size = ???




    def call(self, input_at_t, states_at_t):
        t_direct, t_diffuse, e_split_direct, e_split_diffuse = input_at_t

        r_bottom_direct, r_bottom_diffuse, a_bottom_direct, a_bottom_diffuse = states_at_t

        tmp = propagate_layer_up (t_direct, t_diffuse, e_split_direct, e_split_diffuse, r_bottom_direct, r_bottom_diffuse, a_bottom_direct, a_bottom_diffuse)

        t_multi_direct, t_multi_diffuse, \
            r_multi_direct, r_multi_diffuse, \
            r_bottom_multi_direct, r_bottom_multi_diffuse, \
            a_top_multi_direct, a_top_multi_diffuse, \
            a_bottom_multi_direct, a_bottom_multi_diffuse = tmp

        output_at_t = t_multi_direct, t_multi_diffuse, r_bottom_multi_direct, r_bottom_multi_diffuse
        
        state_at_t_plus_1 = r_multi_direct, r_multi_diffuse, a_top_multi_direct, a_top_multi_diffuse

        return output_at_t, state_at_t_plus_1


class DownwardPropagationCell(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, input_at_t, states_at_t):

        direct_down, diffuse_down = states_at_t

        t_direct, t_diffuse, \
            t_multi_direct, t_multi_diffuse, \
            r_multi_direct, r_multi_diffuse, \
            r_bottom_multi_direct, r_bottom_multi_diffuse, \
            a_top_multi_direct, a_top_multi_diffuse, \
            a_bottom_multi_direct, a_bottom_multi_diffuse = input_at_t

        absorbed_flux = direct_down * a_top_multi_direct + \
                        diffuse_down * a_top_multi_diffuse

        direct_flux_down = direct_down * t_direct
        diffuse_flux_down = direct_down * t_multi_direct + \
                                diffuse_down * (t_diffuse + t_multi_diffuse)
        diffuse_flux_up = direct_down * r_bottom_multi_direct + \
                            diffuse_down * r_bottom_multi_diffuse
        
        output_at_t = absorbed_flux, diffuse_flux_up
        state_at_t_plus_1 = direct_flux_down, diffuse_flux_down

        return output_at_t, state_at_t_plus_1
    
def train():
    n_hidden_gas = [4, 5, 6]
    n_hidden_layer_coefficients = [4, 5, 6]
    n_layers = 60
    n_composition = 8 # 6 gases + liquid water + ice water
    n_channels = 29
    batch_size  = 2048

    # Optical Depth

    # +2 for temp and pressure (at layer not level)
    optical_depth_input = Input(shape=(n_layers, 2 + n_composition), batch_size=batch_size, name="input_input")

    optical_depth = TimeDistributed(OpticalDepth(n_hidden_gas, n_channels), name="optical_depth")(optical_depth_input)

    # Layer coefficients: t, (1-t)t^, (1-t)r^, (1-t)a^

    null_input_1 = Input(shape=(), batch_size=batch_size, name="null_input_1") #could be a ones input
    mu_bar = NoInputLayer(1,name="mu_bar")(null_input_1)
    #mu_bar = NoInputLayer(1,name="mu_bar")() #Not sure how to account for this in model def

    mu_input = Input(shape=(1,), batch_size=batch_size, name="mu")

    # Repeat over all n_channels
    mu_array = tf.expand_dims(mu_input, axis=0)
    mu_array = tf.repeat(mu_array, [n_channels], axis=0)

    optical_depth_and_mu = tf.concat([optical_depth, mu_input, mu_bar], axis=1)

    layer_coefficients = TimeDistributed(LayerProperties(n_hidden_layer_coefficients), name="layer_coefficients")(optical_depth_and_mu)

    # Upward propagation: a and r 

    upward_output, upward_state = tf.keras.layers.RNN(UpwardPropagationCell, return_sequences=True, return_state=False, go_backwards=True)(layer_coefficients)

    # Downward propagation: t and a

    downward_output, downward_state = tf.keras.layers.RNN(DownwardPropagationCell, return_sequences=True, return_state=False, go_backwards=False)(upward_output)

    null_input_2 = Input(shape=(0,), batch_size=batch_size, name="null_input_2")
    channel_split = NoInputLayer(n_nets,name="toa")(null_input_2)
    toa = tf.nn.softmax(channel_split) * 1412.0
    output_1 = tf.multiply(toa, downward_state)
    output_2 = tf.multiply(toa, downward_output)

    model = Model(inputs=[optical_depth_input,null_input_1,mu_input,null_input_2], outputs=[output_1,output_2])

    ###########

    epochs      = 100000
    patience    = 1000 #25

    r_bottom_direct = albedo
    r_bottom_diffuse = albedo
    a_bottom_direct = 1.0 - albedo
    a_bottom_diffuse = 1.0 - albedo

    initial_upward_state = r_bottom_direct, r_bottom_diffuse, a_bottom_direct, a_bottom_diffuse

    t_p_input = Input(shape=(n_layers,2),
                       batch_size=batch_size, name="t_p_input") 
    composition_input = Input(shape=(n_layers,n_composition),
                               batch_size=batch_size, name="composition_input")
    mu_input = Input(shape=(1,), batch_size=batch_size,
                      name="mu_input")
    albedo_input = Input(shape=(2,), batch_size=batch_size,
                          name="albedo_input")
    input = [t_p_input, composition_input, mu_input, albedo_input]
    output = RT_Net(n_hidden_gas, n_hidden_ext)(input)
    model = Model(input, output)

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss="mse")
    callbacks = [EarlyStopping(monitor='rmse_hr',  patience=patience, verbose=1, \
                                 mode='min',restore_best_weights=True)]



    