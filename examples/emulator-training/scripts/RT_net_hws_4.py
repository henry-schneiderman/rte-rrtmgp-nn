import os

import numpy as np
from math import isclose


import tensorflow as tf
from tensorflow.keras import losses, optimizers, layers, Input, Model, Layer, Sequential
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Dense,TimeDistributed


class DenseFFN(Layer):
    """
    n_hidden[n_layers]: array of the number of nodes per layer
    Last layer has RELU activation insuring non-negative output
    """
    def __init__(self, n_hidden, n_outputs):
        super().__init__()
        self.hidden = [Dense(units=n, activation='relu',) for n in n_hidden]
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
        self.gas_index['h2o']=0
        self.gas_index['o3'] =1
        self.gas_index['co2']=2
        self.gas_index['n2o']=3
        self.gas_index['ch4']=4
        # 'u' represents all gases that are horizontally uniform 
        # (but not vertically uniform) in concentration in
        # the atmosphere: o2, n2
        # Could probably include co2, too, but it is separated out for
        # now to account for annual differences
        self.gas_index['uniform']=5  

        self.ke_gas_net = {}

        # Represents a function of temperature and pressure
        # used to build a gas absorption coefficient, ke 

        for gas,n in n_gas.items():
            self.ke_gas_net[gas] = [DenseFFN(n_hidden,1) for _ in np.arange(n)]

        self.ke_lw_net = [NoInputLayer(1) for _ in np.arange(n_channels)]
        self.ke_iw_net = [NoInputLayer(1) for _ in np.arange(n_channels)]

    # Note Ukkonen does not include nitrogen dioxide (no2) in simulation that generated data
    def call(self, input):

        t_p, composition = input

        # Generate multiple optical depths for each gas

        tau_gas = {}
        for gas, ke_gas_net in self.ke_gas_net.items():
            # Extinction coefficient determined by network
            ke = [net(t_p) for net in ke_gas_net]
            # Tau = ke * mass_path_for_gas
            gas_index = self.gas_index[gas]
            tau_gas[gas] = tf.multiply(ke,composition[gas_index])

        h2o = tau_gas['h2o']
        o3 = tau_gas['o3']
        co2 = tau_gas['co2']
        n2o = tau_gas['n2o']
        ch4 = tau_gas['ch4']
        u = tau_gas['uniform']

        # Optical depth for each channel
        # using various combinations of gases' optical depths

        tau_gases = tf.zeros((self._n_channels))
 
        tau_gases[0] = h2o[0] + o3[0] + \
            co2[0] + n2o[0] + ch4[0] + u[0]
        tau_gases[1] = h2o[1] + o3[1] + \
            co2[1] + n2o[1] + ch4[1] + u[1]
        tau_gases[2] = h2o[2] + o3[2] + \
            co2[2] + n2o[2] + ch4[2] + u[2]

        tau_gases[3] = h2o[3] + ch4[3]
        tau_gases[4] = h2o[4] + ch4[4]

        tau_gases[5] = h2o[5] + co2[3]
        tau_gases[6] = h2o[6] + co2[4]

        tau_gases[7] = h2o[3] + ch4[5]
        tau_gases[8] = h2o[4] + ch4[6]

        tau_gases[9]  = h2o[9]  + co2[5]
        tau_gases[10] = h2o[10] + co2[6]

        tau_gases[11] = h2o[11] + ch4[7]
        tau_gases[12] = h2o[12] + ch4[8]

        tau_gases[13] = h2o[13] + co2[7]
        tau_gases[14] = h2o[14] + co2[8]

        tau_gases[15] = h2o[15] + u[3]
        tau_gases[16] = h2o[16] + u[4]

        tau_gases[17] = h2o[17] + o3[3] + u[5]
        tau_gases[18] = h2o[18] + o3[4] + u[6]

        tau_gases[19] = h2o[19] + o3[5] + u[7]
        tau_gases[20] = h2o[20] + o3[6] + u[8]

        tau_gases[21] = h2o[21] + o3[7] + u[9]
        tau_gases[22] = h2o[22] + o3[8] + u[10]

        tau_gases[23] = h2o[23]
        tau_gases[24] = h2o[24]

        tau_gases[25] = h2o[25] + o3[9]
        tau_gases[26] = h2o[26] + o3[10]

        tau_gases[27] = h2o[27] + o3[11] + u[11]
        tau_gases[28] = h2o[28] + o3[12] + u[12]

        # Optical depth for liquid and ice water for each channel
        tau_lw = [net() for net in self.ke_lw_net]
        tau_lw = tf.multiply(tau_lw,composition[6])
        tau_iw = [net() for net in self.ke_iw_net]
        tau_iw = tf.multiply(tau_iw,composition[7])

        return [tau_gases, tau_lw, tau_iw]

class LayerProperties(Layer):
    def __init__(self, n_hidden, n_channels):
        super().__init__()
        self.extinction_net = [DenseFFN(n_hidden,3) for _ in np.arange(n_channels)]

    def call(self, input):

        tau_gases, tau_lw, tau_iw, mu, mu_bar = input

        # Iterate over channels

        e_split_direct = [net(tau_gases[k], tau_lw[k], tau_iw[k], mu[k]) for k, net in self.extinction_net.items()]

        e_split_diffuse = [net(tau_gases[k], tau_lw[k], tau_iw[k], mu_bar[k]) for k, net in self.extinction_net.items()]

        e_split_direct = tf.nn.softmax(e_split_direct)
        e_split_diffuse = tf.nn.softmax(e_split_diffuse)

        # Coefficients of direct transmission of radiation. 
        # Note that diffuse radiation can be directly transmitted

        tau_total = tau_gases + tau_lw + tau_iw

        t_direct = tf.math.exp(-tau_total / mu)

        t_diffuse = tf.math.exp(-tau_total / mu_bar)

        return [t_direct, t_diffuse, e_split_direct, e_split_diffuse]


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
    #
    # Also see Shonk and Hogan, 2007

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
    def __init__(self, n_channels, **kwargs):
        super().__init__(**kwargs)
        self.state_size = (n_channels, 4)
        self.output_size = (n_channels, 4)

    def call(self, input_at_i, states_at_i):
        t_direct, t_diffuse, e_split_direct, e_split_diffuse = input_at_i

        r_bottom_direct, r_bottom_diffuse, a_bottom_direct, a_bottom_diffuse = states_at_i

        tmp = propagate_layer_up (t_direct, t_diffuse, e_split_direct, e_split_diffuse, r_bottom_direct, r_bottom_diffuse, a_bottom_direct, a_bottom_diffuse)

        t_multi_direct, t_multi_diffuse, \
            r_multi_direct, r_multi_diffuse, \
            r_bottom_multi_direct, r_bottom_multi_diffuse, \
            a_top_multi_direct, a_top_multi_diffuse, \
            a_bottom_multi_direct, a_bottom_multi_diffuse= tmp

        output_at_i = [t_multi_direct, t_multi_diffuse, r_bottom_multi_direct, r_bottom_multi_diffuse, 
        a_bottom_multi_direct, a_bottom_multi_diffuse]
        
        state_at_i_plus_1 = [r_multi_direct, r_multi_diffuse, a_top_multi_direct, a_top_multi_diffuse]

        return output_at_i, state_at_i_plus_1


class DownwardPropagationCell(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, input_at_i, states_at_i):

        input_flux_down_direct, input_flux_down_diffuse = states_at_i

        t_direct, t_diffuse, \
        t_multi_direct, t_multi_diffuse, \
        r_bottom_multi_direct, r_bottom_multi_diffuse, \
        a_bottom_multi_direct, a_bottom_multi_diffuse = input_at_i

        #absorbed_flux = input_flux_down_direct * a_top_multi_direct + \
        #                input_flux_down_diffuse * a_top_multi_diffuse

        absorbed_flux_bottom = input_flux_down_direct * a_bottom_multi_direct + \
                                input_flux_down_diffuse * a_bottom_multi_diffuse

        flux_down_direct = input_flux_down_direct * t_direct
        flux_down_diffuse = input_flux_down_direct * t_multi_direct + \
                                input_flux_down_diffuse * (t_diffuse + t_multi_diffuse)
        flux_up_diffuse = input_flux_down_direct * r_bottom_multi_direct + \
                            input_flux_down_diffuse * r_bottom_multi_diffuse
        
        output_at_i = flux_down_direct, flux_down_diffuse, \
            flux_up_diffuse, absorbed_flux_bottom
         
        state_at_i_plus_1 = flux_down_direct, flux_down_diffuse

        return output_at_i, state_at_i_plus_1
    
def train():
    n_hidden_gas = [4, 5, 6]
    n_hidden_layer_coefficients = [4, 5, 6]
    n_layers = 60
    n_composition = 8 # 6 gases + liquid water + ice water
    n_channels = 29
    batch_size  = 2048

    # Optical Depth

    t_p_input = Input(shape=(n_layers,2),
                       batch_size=batch_size, name="t_p_input") 
    
    composition_input = Input(shape=(n_layers,n_composition),
                               batch_size=batch_size, name="composition_input")

    optical_depth = TimeDistributed(OpticalDepth(n_hidden_gas, n_channels), name="optical_depth")([t_p_input, composition_input])

    # Layer coefficients: 
    # direct_transmission, scattered_transmision,
    # scattered_reflection, scattered_absorption


    # This input is always a constant equal to one
    null_mu_bar_input = Input(shape=(0), batch_size=batch_size, name="null_mu_bar_input") 
    mu_bar = Dense(units=1,bias_initializer=tf.constant_initializer(0.5),activation="sigmoid",name="mu_bar")(null_mu_bar_input)

    mu_input = Input(shape=(1), batch_size=batch_size, name="mu_input") 

    layer_properties = TimeDistributed(LayerProperties(n_hidden_layer_coefficients), name="layer_properties")([optical_depth, mu_input, mu_bar])

    # Upward propagation: a and r 
    # (Working upwards layer by layer computing
    # absorption and reflection for the entire atmosphere
    # below the current layer)

    # absorption and reflection (albedo) of the surface
    surface_input = Input(shape=(4), batch_size=batch_size, name="surface_input")

    upward_output, upward_state = tf.keras.layers.RNN(UpwardPropagationCell, return_sequences=True, return_state=True, go_backwards=True)(input=layer_properties, initial_state=surface_input)

    r_multi_direct, r_multi_diffuse, a_top_multi_direct, a_top_multi_diffuse = upward_state

    # Downward propagation:
    # Determine flux absorbed at each level
    # Determine downward and upward flux at all levels

    null_toa_input = Input(shape=(0), batch_size=batch_size, name="null_toa_input")

    top_flux_down_direct = Dense(units=n_channels,bias_initializer='random_normal', activation='softmax')(null_toa_input)

    # Set to zeros
    top_flux_down_diffuse = Input(shape=(n_channels), batch_size=batch_size, name="top_flux_down_diffuse")

    top_flux_up_diffuse = tf.multiply(top_flux_down_direct,r_multi_direct)

    top_absorbed_flux = tf.multiply(top_flux_down_direct, a_top_multi_direct)

    # Downward propagation: t and a
    downward_output = tf.keras.layers.RNN(DownwardPropagationCell, return_sequences=True, return_state=False, go_backwards=True)(input=upward_output, initial_state=[top_flux_down_direct, top_flux_down_diffuse])

    flux_down_direct, flux_down_diffuse, \
            flux_up_diffuse, absorbed_flux_bottom = downward_output

    flux_down_direct = tf.concat([top_flux_down_direct,flux_down_direct], axis=0)

    flux_down_diffuse = tf.concat([top_flux_down_diffuse,flux_down_diffuse], axis=0)

    absorbed_flux = tf.concat([top_absorbed_flux, absorbed_flux_bottom], axis=0)

    flux_up_diffuse = tf.concat([top_flux_up_diffuse, flux_up_diffuse], axis=0)

    model = Model(inputs=[optical_depth_input,null_input_1,mu_input,null_input_2, surface_input, toa_input], outputs=[output_1,output_2])

    ###########

    epochs      = 100000
    patience    = 1000 #25

    r_bottom_direct = albedo
    r_bottom_diffuse = albedo
    a_bottom_direct = 1.0 - albedo
    a_bottom_diffuse = 1.0 - albedo

    initial_upward_state = r_bottom_direct, r_bottom_diffuse, a_bottom_direct, a_bottom_diffuse

    t_down_direct = 1412.0
    t_down_diffuse = 0.0

    t_up_diffuse = t_down_direct + r_multi_direct + t_down_diffuse * r_multi_diffuse


    input = [t_p_input, composition_input, mu_input, albedo_input]
    output = RT_Net(n_hidden_gas, n_hidden_ext)(input)
    model = Model(input, output)

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss="mse")
    callbacks = [EarlyStopping(monitor='rmse_hr',  patience=patience, verbose=1, \
                                 mode='min',restore_best_weights=True)]



    