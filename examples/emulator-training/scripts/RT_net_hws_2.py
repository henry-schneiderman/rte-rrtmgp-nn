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
        return(self.w)

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
class AtmLayer(Layer):
    def __init__(self, n_hidden_gas, n_hidden_ext):
        super().__init__()

        self._n_nets=29

        n_gas = {}
        n_gas['h2o']=23
        n_gas['h2o_squared']=29
        n_gas['o3'] =13
        n_gas['co2']=9
        n_gas['n2o']=3
        n_gas['ch4']=9
        # 'u' represents all gases that are uniform in concentration in
        # the atmosphere: o2, n2
        # Could probably include co2, too, but it is separated out for
        # now to account for annual differences
        n_gas['u']=13  

        self.ke_gas = {}

        # Represents a function of temperature and pressure
        # used to build a gas absorption coefficient, ke 

        for gas,n in n_gas.items():
            self.ke_gas[gas] = [DenseFFN(n_hidden_gas,1) for _ in np.arange(n)]

        n_lw=self._n_nets
        n_iw=self._n_nets

        self.ke_lw = [NoInputLayer(1) for _ in np.arange(n_lw)]
        self.ke_iw = [NoInputLayer(1) for _ in np.arange(n_iw)]

        self.ext_net = [DenseFFN(n_hidden_ext,3) for _ in np.arange(self._n_nets)]

    # Note Ukkonen does not include nitrogen dioxide (no2) in simulation that generated data
    def call(self, input):
        temp, pressure, composition, lwp, iwp, mu, mu_bar = input
        X = tf.concat([temp,pressure],axis=-1)

        tau_gas = {}
        for gas, ke_gas in self.ke_gas.items():
            tau_gas[gas] = [net(X) for net in ke_gas]
            tau_gas[gas] = tf.multiply(tau_gas[gas],composition[gas])

        tau_lw = [net() for net in self.ke_lw]
        tau_lw = tf.multiply(tau_lw,lwp)
        tau_iw = [net() for net in self.ke_iw]
        tau_iw = tf.multiply(tau_iw,iwp)

        h2o = tau_gas['h2o']
        h2o_sq = tau_gas['h2o_sq']
        o3 = tau_gas['o3']
        co2 = tau_gas['co2']
        n2o = tau_gas['n2o']
        ch4 = tau_gas['ch4']
        u = tau_gas['uniform']

        tau_gases = tf.zeros((self._n_nets,1))
 
        tau_gases[0,0] = h2o[0] + o3[0] + co2[0] + n2o[0] + ch4[0] + u[0] + h2o_sq[0]
        tau_gases[1,0] = h2o[1] + o3[1] + co2[1] + n2o[1] + ch4[1] + u[1] + h2o_sq[1]
        tau_gases[2,0] = h2o[2] + o3[2] + co2[2] + n2o[2] + ch4[2] + u[2] + h2o_sq[2]

        tau_gases[3,0] = h2o[3] + ch4[3] + h2o_sq[3]
        tau_gases[4,0] = h2o[4] + ch4[4] + h2o_sq[4]

        tau_gases[5,0] = h2o[5] + co2[3] + h2o_sq[5]
        tau_gases[6,0] = h2o[6] + co2[4] + h2o_sq[6]

        tau_gases[7,0] = h2o[3] + ch4[5] + h2o_sq[7]
        tau_gases[8,0] = h2o[4] + ch4[6] + h2o_sq[8]

        tau_gases[9,0]  = h2o[9]  + co2[5] + h2o_sq[9]
        tau_gases[10,0] = h2o[10] + co2[6] + h2o_sq[10]

        tau_gases[11,0] = h2o[11] + ch4[7] + h2o_sq[11]
        tau_gases[12,0] = h2o[12] + ch4[8] + h2o_sq[12]

        tau_gases[13,0] = h2o[13] + co2[7] + h2o_sq[13]
        tau_gases[14,0] = h2o[14] + co2[8] + h2o_sq[14]

        tau_gases[15,0] = h2o[15] + u[3] + h2o_sq[15]
        tau_gases[16,0] = h2o[16] + u[4] + h2o_sq[16]

        tau_gases[17,0] = h2o[17] + o3[3] + u[5] + h2o_sq[17]
        tau_gases[18,0] = h2o[18] + o3[4] + u[6] + h2o_sq[18]

        tau_gases[19,0] = h2o[19] + o3[5] + u[7] + h2o_sq[19]
        tau_gases[20,0] = h2o[20] + o3[6] + u[8] + h2o_sq[20]

        tau_gases[21,0] = h2o[21] + o3[7] + u[9] + h2o_sq[21]
        tau_gases[22,0] = h2o[22] + o3[8] + u[10] + h2o_sq[22]

        tau_gases[23,0] = h2o_sq[23]
        tau_gases[24,0] = h2o_sq[24]

        tau_gases[25,0] = o3[9] + h2o_sq[25]
        tau_gases[26,0] = o3[10] + h2o_sq[26]

        tau_gases[27,0] = o3[11] + u[11] + h2o_sq[27]
        tau_gases[28,0] = o3[12] + u[12] + h2o_sq[28]

        tau_total = tau_gases + tau_lw + tau_iw
        input_1 = tau_lw / tau_total
        input_2 = tau_iw / tau_total

        input_3_direct = tau_total / mu
        input_3_diffuse = tau_total / mu_bar

        e_direct_split = [net(input_1[k],input_2[k],input_3_direct[k], mu) for k, net in self.ext_net.items()]
        e_diffuse_split = [net(input_1[k],input_2[k],input_3_diffuse[k], mu_bar) for k, net in self.ext_net.items()]

        e_direct_split = tf.nn.softmax(e_direct_split)
        e_diffuse_split = tf.nn.softmax(e_diffuse_split)

        t_direct = tf.exp(-input_3_direct)
        t_diffuse = tf.exp(-input_3_diffuse)

        return t_direct, t_diffuse, e_direct_split, e_diffuse_split
    
def propagate_layer (t_direct, t_diffuse, e_direct_split, e_diffuse_split, r_bottom_direct, a_bottom_direct, r_bottom_diffuse, a_bottom_diffuse):
    """
    Computes the downward total absorption and reflection coefficients for a column of the atmosphere
    from the given layer to the surface. 

    Uses the atmospheric properties of the given "top layer" and the total absorption and reflection 
    of the "bottom layer" spanning all the layers beneath the top layer including the surface. 
    Computes the impact of multi-reflection between these top and bottom layers.

    Input and Output Shape:
        Tensor with shape (n_batches, n_channels)

    Arguments:
        t_direct (n_batches,n_channels) - Direct transmission coefficient for direct radiation 
            passing through the top layer
        t_diffuse (n_batches,n_channels) - Direct transmission coefficient for diffuse radiation 
            passing through the top layer
        e_direct_split - The split of the extinguished direct radiation 
            (direct radiation that is not directly transmitted) into diffusely transmitted,
            reflected and absorbed components. Has additional axis of length=3.
        e_diffuse_split - The split of the extinguished diffuse radiation 
            (diffuse radiation that is not directly transmitted) into diffusely transmitted,
            reflected and absorbed components.  Has additional axis of length=3.
        r_bottom_direct - The total reflection coefficient for the bottom layer of 
            direct downward radiation.
        a_bottom_direct - The total absorption coefficient for the bottom layer of 
            direct downward radiation.
        r_bottom_diffuse - The total reflection coefficient for the bottom layer of 
            diffuse downward radiation.
        a_bottom_diffuse - The total absorption coefficient for the bottom layer of 
            diffuse downward radiation.

    Returns:
        r_multi_direct, r_multi_diffuse - The total effective reflection coefficient 
            for the combined top and bottom layer including the surface
        a_multi_direct, a_multi_diffuse - The absorption coefficients of the top layer after 
            multi-reflection between the layers
        a_bottom_multi_direct, a_bottom_multi_diffuse - The absorption coefficients 
            of the lower layer after multi-reflection between the layers

    Notes:
        Conservation of energy:
            a_bottom_multi_direct + a_multi_direct + r_multi_direct = 1.0
            a_bottom_multi_diffuse + a_multi_diffuse + r_multi_diffuse = 1.0

    """
    # The top layer splits the direct beam into transmitted and extinguished components
    e_direct = 1 - t_direct
    
    # The top layer also splits the downward diffuse flux into transmitted and extinguished components
    e_diffuse = 1 - t_diffuse

    # The top layer further splits each extinguished component into transmitted, reflected,
    # and absorbed components
    e_t_direct, e_r_direct, e_a_direct = tf.transpose(e_direct_split, perm=[2,0,1])
    e_t_diffuse, e_r_diffuse, e_a_diffuse = tf.transpose(e_diffuse_split, perm=[2,0,1])

    # Multi-reflection between the top layer and lower layer resolves 
    # a direct beam into:
    #   r_multi_direct - total effective reflection at the top layer
    #   a_multi_direct - absorption at the top layer
    #   a_bottom_multi_direct - absorption for the entire bottom layer

    # The adding-doubling method computes these
    # See p.418-424 of "A First Course in Atmospheric Radiation (2nd edition)"
    # by Grant W. Petty

    d = 1.0 / (1.0 - e_diffuse * e_r_diffuse * r_bottom_diffuse)

    a_bottom_multi_direct = t_direct * a_bottom_direct + \
        t_direct * r_bottom_direct * e_diffuse * e_r_diffuse * a_bottom_diffuse * d + \
        e_direct * e_t_direct * a_bottom_diffuse * d

    a_multi_direct = e_direct * e_a_direct + \
        t_direct * r_bottom_direct * e_diffuse * e_a_diffuse * d + \
        e_direct * e_t_direct * r_bottom_diffuse * e_diffuse * e_a_diffuse * d

    r_multi_direct = e_direct * e_r_direct + \
        t_direct * r_bottom_direct * t_diffuse * d + \
        t_direct * r_bottom_direct * e_diffuse * e_t_diffuse * d + \
        e_direct * e_t_direct * r_bottom_diffuse * t_diffuse * d + \
        e_direct * e_t_direct * r_bottom_diffuse * e_diffuse * e_t_diffuse * d

    # These should sum to 1.0
    total_direct = a_bottom_multi_direct + a_multi_direct + r_multi_direct
    assert isclose(total_direct, 1.0, abs_tol=1e-5)

    # Multi-reflection for diffuse flux
    a_bottom_multi_diffuse = t_diffuse * a_bottom_diffuse + \
        t_diffuse * r_bottom_diffuse * e_diffuse * e_r_diffuse * a_bottom_diffuse * d + \
        e_diffuse * e_t_diffuse * a_bottom_diffuse * d

    a_multi_diffuse = e_diffuse * e_a_diffuse + \
        t_diffuse * r_bottom_diffuse * e_diffuse * e_a_diffuse * d + \
        e_diffuse * e_t_diffuse * r_bottom_diffuse * e_diffuse * e_a_diffuse * d

    r_multi_diffuse = e_diffuse * e_r_diffuse + \
        t_diffuse * r_bottom_diffuse * t_diffuse * d + \
        t_diffuse * r_bottom_diffuse * e_diffuse * e_t_diffuse * d + \
        e_diffuse * e_t_diffuse * r_bottom_diffuse * t_diffuse * d + \
        e_diffuse * e_t_diffuse * r_bottom_diffuse * e_diffuse * e_t_diffuse * d

    total_diffuse = a_bottom_multi_diffuse + a_multi_diffuse + r_multi_diffuse
    assert isclose(total_diffuse, 1.0, abs_tol=1e-5)

    return r_multi_direct, a_multi_direct, a_bottom_multi_direct, \
            r_multi_diffuse, a_multi_diffuse, a_bottom_multi_diffuse

class RT_Net(Layer):
    def __init__(self, n_hidden_gas, n_hidden_ext):
        super().__init__()
        self.layer=AtmLayer(n_hidden_gas=n_hidden_gas, n_hidden_ext=n_hidden_ext)
        self.mu_bar=NoInputLayer(1)
    def call(self, input):
        layer_input, mu, surface_albedo = input
        mu_bar = self.mu_bar()
        r_bottom_direct = surface_albedo
        r_bottom_diffuse = surface_albedo
        a_bottom_direct = 1.0 - surface_albedo
        a_bottom_diffuse = 1.0 - surface_albedo
        input_shape = layer_input.shape
        composite_coefficients = tf.zeros((input_shape[0],input_shape[1],6))
        for i in np.arange(input_shape[0]):
            layer_coefficients=self.layer(layer_input[i])
            t_direct, t_diffuse, e_direct_split, e_diffuse_split = layer_coefficients
            composite_coefficients[i]= propagate_layer(t_direct, t_diffuse, e_direct_split, e_diffuse_split, r_bottom_direct, a_bottom_direct, r_bottom_diffuse, a_bottom_diffuse)
            r_multi_direct, a_multi_direct, a_bottom_multi_direct, \
            r_multi_diffuse, a_multi_diffuse, a_bottom_multi_diffuse = composite_coefficients[i]
            r_bottom_direct = r_multi_direct
            r_bottom_diffuse = r_multi_diffuse
            a_bottom_direct = a_multi_direct + a_bottom_multi_direct
            a_bottom_diffuse = a_multi_diffuse + a_bottom_multi_diffuse

def train():
    epochs      = 100000
    patience    = 1000 #25
    batch_size  = 2048
    layer_input = Input(shape(n_layers,n_features), batch_size=batch_size,   name="layer_input")
    mu_input = Input(shape=(1,), batch_size=batch_size, name="mu_input")
    albedo_input = Input(shape=(2,), batch_size=batch_size, name="albedo_input")
    input = [layer_input, mu_input, albedo_input]
    output = RT_Net(n_hidden_gas, n_hidden_ext)(input)
    model = Model(input, output)

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss="mse")
    callbacks = [EarlyStopping(monitor='rmse_hr',  patience=patience, verbose=1, \
                                 mode='min',restore_best_weights=True)]

