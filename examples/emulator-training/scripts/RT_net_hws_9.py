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

from RT_data_hws import load_data_direct, load_data_full, absorbed_flux_to_heating_rate

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.config.optimizer.set_jit(True)

class DenseFFN(Layer):
    """
    n_hidden[n_layers]: array of the number of nodes per layer
    Last layer has RELU activation insuring non-negative output
    """
    def __init__(self, n_hidden, n_outputs, minval, maxval, **kwargs):
        super().__init__(**kwargs)
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.minval = minval
        self.maxval = maxval
        l2_regularization = 0.000

        self.hidden = [Dense(units=n, activation='relu',kernel_initializer=initializers.RandomUniform(minval=minval, maxval=maxval),                         
                             #kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
                            bias_initializer=initializers.RandomNormal(mean=1.0, stddev=0.05)) for n in n_hidden]
        
        #self.batch_normalization = [tf.keras.layers.BatchNormalization(**kwargs) for _ in n_hidden]

        self.dropout = [tf.keras.layers.Dropout(0.0, **kwargs) for _ in n_hidden]  #0.15, 0.075, 0.0375
        # Sigmoid insures that output is non-negative
        # Sigmoid with input of 0.0 gives output of 0.5
        self.out = Dense(units=n_outputs, activation='sigmoid',kernel_initializer=initializers.RandomUniform(minval=minval, maxval=maxval), bias_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05)) 

    def call(self, X, training=False):
        for k, hidden in enumerate(self.hidden):
            X = hidden(X)
            X = self.dropout[k](X,training=training)
            #X = tf.nn.relu(self.batch_normalization[k](X, training=training))
        return self.out(X)
    
            
    def get_config(self):
        base_config = super(DenseFFN, self).get_config()
        config = {
            'n_hidden': self.n_hidden,
            'n_outputs': self.n_outputs,
            'minval': self.minval,
            'maxval': self.maxval,
        }
        return config.update(base_config)
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ScatteringNet(Layer):
    """ Computes split of extinguished radiation into absorbed, diffuse transmitted, and
    diffuse reflected """
    def __init__(self, **kargs):
        super().__init__(**kargs)
        self.n_hidden = [5, 4, 4]
        
        self.hidden_net = [Dense(n_hidden,activation=tf.keras.activations.relu,  
                        kernel_initializer=tf.keras.initializers.glorot_uniform()) for n_hidden in self.n_hidden]
        
        self.output_net = Dense(units=3, 
                                activation=tf.keras.activations.softmax,kernel_initializer=tf.keras.initializers.glorot_uniform())
        
    def call(self, X, training=False):

        for k, hidden in enumerate(self.hidden_net):
            X = hidden(X)
            #X = tf.nn.relu(self.batch_normalization[k](X, training=training))
        return self.output_net(X)
    
            
    def get_config(self):
        base_config = super(ScatteringNet, self).get_config()
        config = {
        }
        return config.update(base_config)
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class OpticalDepth(Layer):
    def __init__(self, n_channels, **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.n_o3 = 13 
        self.n_co2 = 9 
        self.n_u = 13
        self.n_n2o = 3
        self.n_ch4 = 9
        l2_regularization = 0.0001

        self.net_lw = Dense(units=self.n_channels,
                        activation=tf.keras.activations.relu,  
                        name = 'net_lw',  
                        #kernel_constraint=tf.keras.constraints.NonNeg(),
                        #kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
                        kernel_initializer=initializers.RandomUniform(minval=0.38, maxval=1.24), #maxval=0.62
                        use_bias=False)
                        #kernel_initializer=initializers.RandomUniform(minval=0.10, maxval=1.0),use_bias=False)

        self.net_iw = Dense(units=self.n_channels,
                        activation=tf.keras.activations.relu,
                        name = 'net_iw',         
                        #kernel_constraint=tf.keras.constraints.NonNeg(),  
                        #kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
                        kernel_initializer=initializers.RandomUniform(minval=0.38, maxval=1.24), #maxval=0.62
                        use_bias=False)

        self.net_h2o = Dense(units=self.n_channels,
                        activation=tf.keras.activations.relu,   
                        name = 'net_h2o',
                        #kernel_constraint=tf.keras.constraints.NonNeg(),
                        #kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
                        kernel_initializer=initializers.RandomUniform(minval=0.38, maxval=1.24), #maxval=0.62
                        use_bias=False)

        self.net_h2o_sq = Dense(units=1, #self.n_channels,
                        activation=tf.keras.activations.relu,   
                        name = 'net_h2o_sq',
                        #kernel_constraint=tf.keras.constraints.NonNeg(), 
                        #kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
                        kernel_initializer=initializers.RandomUniform(minval=0.38, maxval=1.24), #maxval=0.62
                        use_bias=False)
        """
        self.net_ke_h2o = Dense(units=self.n_channels, # 1,
                        #activation=tf.keras.activations.relu,
                        activation=tf.keras.activations.sigmoid,
                        name = 'net_ke_h2o',         
                        use_bias=True,                  
                        kernel_initializer='zeros',
                        bias_initializer='zeros')
        """


        #self.n_channels, 
        self.net_ke_h2o = DenseFFN(n_hidden=(6,4,4), n_outputs=1, minval=-0.1, maxval=0.1, name='net_ke_h2o')

        self.net_ke_h2o_sq = DenseFFN(n_hidden=(6,4,4), n_outputs=1, minval=-0.1, maxval=0.1, name='net_ke_h2o_sq')
        
        self.net_o3 = Dense(units=self.n_o3,
                        activation=tf.keras.activations.relu,  
                        name = 'net_o3',                                 
                        #kernel_constraint=tf.keras.constraints.NonNeg(),
                        #kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
                        kernel_initializer=initializers.RandomUniform(minval=0.38, maxval=1.24), #maxval=0.62
                        use_bias=False)
        """
        self.net_ke_o3 = Dense(units=self.n_o3,  #1,
                        #activation=tf.keras.activations.relu,
                        activation=tf.keras.activations.sigmoid,                     
                        #kernel_initializer=initializers.RandomUniform(minval=0.0001, maxval=1.0),use_bias=True)
                        name = 'net_ke_o3',     
                        use_bias=True,
                        kernel_initializer='zeros', bias_initializer='zeros')
        """

        self.net_ke_o3 = DenseFFN(n_hidden=(6,4,4), n_outputs=1, minval=-0.1, maxval=0.1, name='net_ke_o3')

        self.net_co2 = Dense(units=self.n_co2,
                        activation=tf.keras.activations.relu, 
                        name = 'net_co2',      
                        #kernel_constraint=tf.keras.constraints.NonNeg(),  
                        #kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
                        kernel_initializer=initializers.RandomUniform(minval=0.38, maxval=1.24), #maxval=0.62
                        use_bias=False)
        
        """
        self.net_ke_co2 = Dense(units=self.n_co2,  #1,
                        #activation=tf.keras.activations.relu,
                        activation=tf.keras.activations.sigmoid,                     
                        #kernel_initializer=initializers.RandomUniform(minval=0.0001, maxval=1.0),use_bias=True)
                        name = 'net_ke_co2',     
                        use_bias=True,
                        kernel_initializer='zeros', bias_initializer='zeros')
        """

        self.net_ke_co2 = DenseFFN(n_hidden=(6, 4,4), n_outputs=1, minval=-0.1, maxval=0.1, name='net_ke_co2')

        self.net_u = Dense(units=self.n_u,
                        activation=tf.keras.activations.relu,    
                        name = 'net_u',  
                        #kernel_constraint=tf.keras.constraints.NonNeg(), 
                        #kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
                        kernel_initializer=initializers.RandomUniform(minval=0.38, maxval=1.24), #maxval=0.62
                        use_bias=False)
        
        """
        self.net_ke_u = Dense(units=self.n_u,   #1,
                        #activation=tf.keras.activations.relu,
                        activation=tf.keras.activations.sigmoid,                     
                        #kernel_initializer=initializers.RandomUniform(minval=0.0001, maxval=1.0),use_bias=True)
                        name = 'net_ke_u',     
                        use_bias=True,
                        kernel_initializer='zeros', bias_initializer='zeros')
        """

        self.net_ke_u = DenseFFN(n_hidden=(6, 4,4), n_outputs=1, minval=-0.1, maxval=0.1, name='net_ke_u')

        self.net_n2o = Dense(units=self.n_n2o,
                        activation=tf.keras.activations.relu,       
                        name = 'net_n2o',      
                        #kernel_constraint=tf.keras.constraints.NonNeg(), 
                        #kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
                        kernel_initializer=initializers.RandomUniform(minval=0.38, maxval=1.24), #maxval=0.62
                        use_bias=False)

        """
        self.net_ke_n2o = Dense(units=self.n_n2o,   #1,
                        #activation=tf.keras.activations.relu,
                        activation=tf.keras.activations.sigmoid,                     
                        #kernel_initializer=initializers.RandomUniform(minval=0.0001, maxval=1.0),use_bias=True)
                        name = 'net_ke_n2o',     
                        use_bias=True,
                        kernel_initializer='zeros', bias_initializer='zeros')
        """

        self.net_ke_n2o = DenseFFN(n_hidden=(6, 4,4), n_outputs=1, minval=-0.1, maxval=0.1, name='net_ke_n2o')

        self.net_ch4 = Dense(units=self.n_ch4,
                        activation=tf.keras.activations.relu, 
                        name = 'net_ch4',          
                        #kernel_constraint=tf.keras.constraints.NonNeg(), 
                        #kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
                        kernel_initializer=initializers.RandomUniform(minval=0.38, maxval=1.24), #maxval=0.62
                        use_bias=False)
        
        """
        self.net_ke_ch4 = Dense(units=self.n_ch4, #1,
                        #activation=tf.keras.activations.relu,
                        activation=tf.keras.activations.sigmoid,                     
                        #kernel_initializer=initializers.RandomUniform(minval=0.0001, maxval=1.0),use_bias=True)
                        name = 'net_ke_ch4',     
                        use_bias=True,
                        kernel_initializer='zeros', bias_initializer='zeros')
        """

        self.net_ke_ch4 = DenseFFN(n_hidden=(6, 4,4), n_outputs=1, minval=-0.1, maxval=0.1, name='net_ke_ch4')


    def call(self, input, training=False):

        lw, h2o, o3, co2, u, n2o, ch4, h2o_sq, t_p = input

        tau_lw = self.net_lw(lw[:,0:1], training=training)
        tau_iw = self.net_iw(lw[:,1:2], training=training)

        ke_h2o = self.net_ke_h2o(t_p, training=training) 
        tau_h2o = ke_h2o * self.net_h2o(h2o, training=training)

        ke_h2o_sq = self.net_ke_h2o_sq(t_p, training=training) 
        tau_h2o_sq = ke_h2o_sq * self.net_h2o_sq(h2o_sq, training=training)

        ke_o3 = self.net_ke_o3(t_p, training=training)
        tau_o3 = ke_o3 * self.net_o3(o3, training=training)
        #self.n_o3 = 13 
        # amount of padding on each side
        paddings = tf.constant([[0,0],[0,self.n_channels - self.n_o3]])
        tau_o3 = tf.pad(tau_o3, paddings, "CONSTANT")

        ke_co2 = self.net_ke_co2(t_p, training=training)
        tau_co2 = ke_co2 * self.net_co2(co2, training=training) 
        #self.n_co2 = 9 
        #overlaps o3 by 3 and no-overlap for 6
        paddings = tf.constant([[0,0],[self.n_o3 - 3, self.n_channels - ((self.n_o3 - 3) + self.n_co2)]])
        tau_co2 = tf.pad(tau_co2, paddings, "CONSTANT")

        ke_u = self.net_ke_u(t_p, training=training)
        tau_u = ke_u * self.net_u(u, training=training) 
        # self.n_u = 13
        # 5 channels
        # overlap with o3 only (2) and o3 + co2 (3)
        paddings_1 = tf.constant([[0,0],[self.n_o3 - 5, self.n_channels - ((self.n_o3 - 5) + 5)]])
        tau_u_1 = tf.pad(tau_u[:,:5], paddings_1, "CONSTANT")
        # Remaining 8 channels: no overlap with o3 or co2
        paddings_2 = tf.constant([[0,0],[(self.n_o3 - 3) + self.n_co2, self.n_channels - ((self.n_o3 - 3) + self.n_co2 + 8)]])
        tau_u_2 = tf.pad(tau_u[:,5:], paddings_2, "CONSTANT")
        tau_u = tau_u_1 + tau_u_2

        ke_n2o = self.net_ke_n2o(t_p, training=training)
        tau_n2o = ke_n2o * self.net_n2o(n2o, training=training) 
        #self.n_n2o = 3
        #overlaps everything by 3
        paddings = tf.constant([[0,0],[self.n_o3 - 3, self.n_channels - ((self.n_o3 - 3) + self.n_n2o)]])
        tau_n2o = tf.pad(tau_n2o, paddings, "CONSTANT")

        ke_ch4 = self.net_ke_ch4(t_p, training=training)
        tau_ch4 = ke_ch4 * self.net_ch4(ch4, training=training) 
        #self.n_ch4 = 9
        #overlaps everything by 3
        paddings_a = tf.constant([[0,0],[self.n_o3 - 3, self.n_channels - ((self.n_o3 - 3) + 3)]])
        tau_ch4_1 = tf.pad(tau_ch4[:,:3], paddings_a, "CONSTANT")

        #overlap o2 by 2, no overlap for 2
        paddings_b = tf.constant([[0,0],[self.n_channels - 4, 0]])
        tau_ch4_2 = tf.pad(tau_ch4[:,3:7], paddings_b, "CONSTANT")

        #overlap o3 by 2
        paddings_c = tf.constant([[0,0],[0, self.n_channels - 2]])
        tau_ch4_3 = tf.pad(tau_ch4[:,7:], paddings_c, "CONSTANT")
        tau_ch4 = tau_ch4_1 + tau_ch4_2 + tau_ch4_3

        tau_lw = tf.expand_dims(tau_lw, axis=2)
        tau_iw = tf.expand_dims(tau_iw, axis=2)

        if False:
            tau_gases = tau_h2o + tau_o3 + tau_co2 + tau_u + tau_n2o + tau_ch4 + tau_h2o_sq
            tau_gases = tf.expand_dims(tau_gases, axis=2)
            tau = tf.concat((tau_lw, tau_iw, tau_gases), axis=2)
        else:
            tau_h2o = tf.expand_dims(tau_h2o, axis=2)
            tau_o3 = tf.expand_dims(tau_o3, axis=2)
            tau_co2 = tf.expand_dims(tau_co2, axis=2)
            tau_u = tf.expand_dims(tau_u, axis=2)
            tau_n2o = tf.expand_dims(tau_n2o, axis=2)
            tau_ch4 = tf.expand_dims(tau_ch4, axis=2)
            tau_h2o_sq = tf.repeat(tau_h2o_sq, self.n_channels, axis=1)
            tau_h2o_sq = tf.expand_dims(tau_h2o_sq, axis=2)
            tau = tf.concat((tau_lw, tau_iw, tau_h2o, tau_o3, tau_co2, tau_u, tau_n2o, tau_ch4, tau_h2o_sq), axis=2)

        return tau

    
    def compute_output_shape(self, input_shape):
        return [tf.TensorShape([input_shape[0][0],self.n_channels,1])]
        #return [tf.TensorShape([input_shape[0][0],self.n_channels,1]), tf.TensorShape([input_shape[0][0],self.n_channels,1]), tf.TensorShape([input_shape[0][0],self.n_channels,1])]
        
    def get_config(self):
        base_config = super(OpticalDepth, self).get_config()
        config = {
            'n_channels': self.n_channels,
        }
        return config.update(base_config)
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class LayerPropertiesDirect(Layer):
    """ Only Comutes Direct Transmission Coefficient """
    def __init__(self, **kargs):
        super().__init__(**kargs)

    def call(self, input, **kargs):

        tau, mu, u = input

        #print(f"LayerProperties(): shape of taus = {tau.shape}")

        mu = tf.expand_dims(mu, axis=2)
        tau_total = tf.reduce_sum(tau, axis=-1, keepdims=True)

        #print(f'Shape of mu = {mu.shape}')
        #print(" ")

        t_direct = tf.math.exp(-tau_total / (mu + 0.0000001))

        #u = tf.expand_dims(u,axis=1)
        #t_direct = tf.math.exp(-(tau_total * u) / (mu + 0.0000001))

        #print(f'Shape of t_direct = {t_direct.shape}')
        #print(" ")

        return t_direct

    def get_config(self):
        base_config = super(LayerPropertiesDirect, self).get_config()
        config = {
        }
        return config.update(base_config)
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class LayerProperties(Layer):
    """ Computes split of extinguished radiation into absorbed, diffuse transmitted, and
    diffuse reflected """
    def __init__(self, **kargs):
        super().__init__(**kargs)
        self.n_hidden = [5, 4, 4]
        self.input_net_direct = Dense(units=self.n_hidden[0],
                        activation=tf.keras.activations.elu,  
                        kernel_initializer=tf.keras.initializers.glorot_uniform())
        
        self.hidden_net_direct = [Dense(n_hidden,activation=tf.keras.activations.elu,  
                        kernel_initializer=tf.keras.initializers.glorot_uniform()) for n_hidden in self.n_hidden[1:]]
        
        self.output_net_direct = Dense(units=3, 
                                activation=tf.keras.activations.softmax,kernel_initializer=tf.keras.initializers.glorot_uniform())

        self.input_net_diffuse = Dense(units=self.n_hidden[0],
                        activation=tf.keras.activations.elu,  
                        kernel_initializer=tf.keras.initializers.glorot_uniform())
        
        self.hidden_net_diffuse = [Dense(n_hidden,activation=tf.keras.activations.elu,  
                        kernel_initializer=tf.keras.initializers.glorot_uniform()) for n_hidden in self.n_hidden[1:]]
        
        self.output_net_diffuse = Dense(units=3, 
                                activation=tf.keras.activations.softmax,kernel_initializer=tf.keras.initializers.glorot_uniform())

    def call(self, input, **kargs):

        # tau.shape = (n, 29, n_constituents)
        tau, mu, mu_bar = input

        #print(f"LayerProperties(): shape of taus = {tau.shape}")

        mu = tf.expand_dims(mu, axis=2)
        mu_bar = tf.expand_dims(mu_bar, axis=2)

        x_direct = self.input_net_direct(tau / mu)
        x_diffuse = self.input_net_diffuse(tau / mu_bar)

        for net in self.hidden_net_direct:
            x_direct = net(x_direct, **kargs)

        for net in self.hidden_net_diffuse:
            x_diffuse = net(x_diffuse, **kargs)

        e_split_direct = self.output_net_direct(x_direct, **kargs)
        e_split_diffuse = self.output_net_diffuse(x_diffuse, **kargs)

        #print(f'Shape of e_split_diffuse = {e_split_diffuse.shape}')
        #print(" ")

        # Coefficients of direct transmission of radiation. 
        # Note that diffuse radiation can be directly transmitted

        tau_total = tf.reduce_sum(tau, axis=-1, keepdims=True)

        #print(f'Shape of mu = {mu.shape}')
        #print(" ")

        #print(f'Shape of mu_bar = {mu_bar.shape}')
        #print(" ")


        t_direct = tf.math.exp(-tau_total / (mu + 0.0000001))
        t_diffuse = tf.math.exp(-tau_total / (mu_bar + 0.0000001))

        #print(f'Shape of t_direct = {t_direct.shape}')
        #print(" ")

        #e_split_direct = tf.transpose(e_split_direct,perm=[1,0,2])
        #e_split_diffuse = tf.transpose(e_split_diffuse,perm=[1,0,2])

        #print(f'Shape of e_split_diffuse = {e_split_diffuse.shape}')
        #print(" ")

        layer_properties = tf.concat([t_direct, t_diffuse, e_split_direct, e_split_diffuse], axis=2)

        return layer_properties
    

    def get_config(self):
        base_config = super(LayerProperties, self).get_config()
        config = {
        }
        return config.update(base_config)
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class LayerPropertiesScattering(Layer):
    """ Computes split of extinguished radiation into absorbed, diffuse transmitted, and
    diffuse reflected """
    def __init__(self, n_channels, **kargs):
        super().__init__(**kargs)
        
        self.n_channels = n_channels

        self.net_direct = [ScatteringNet() for _ in range(self.n_channels)]
        
        self.net_diffuse = [ScatteringNet() for _ in range(self.n_channels)]

    def call(self, input, **kargs):

        # tau.shape = (n, 29, n_constituents)
        tau, mu, mu_bar, lw, h2o, o3, co2, u, n2o, ch4 = input

        constituents = tf.concat([lw, h2o, o3, co2, u, n2o, ch4], axis=1)

        constituents_direct = constituents / (mu + 0.0000001)
        constituents_diffuse = constituents / (mu_bar + 0.0000001)

        mu = tf.expand_dims(mu, axis=2)
        mu_bar = tf.expand_dims(mu_bar, axis=2)

        tau_total = tf.reduce_sum(tau, axis=-1, keepdims=True)

        t_direct = tf.math.exp(-tau_total / (mu + 0.0000001))
        t_diffuse = tf.math.exp(-tau_total / (mu_bar + 0.0000001))

        e_split_direct_list = [net(constituents_direct) for net in self.net_direct]
        e_split_diffuse_list = [net(constituents_diffuse) for net in self.net_diffuse]

        e_split_direct = tf.convert_to_tensor(e_split_direct_list)
        e_split_diffuse = tf.convert_to_tensor(e_split_diffuse_list)

        e_split_direct = tf.transpose(e_split_direct,perm=[1,0,2])
        e_split_diffuse = tf.transpose(e_split_diffuse,perm=[1,0,2])

        layer_properties = tf.concat([t_direct, t_diffuse, e_split_direct, e_split_diffuse], axis=2)

        return layer_properties
    

    def get_config(self):
        base_config = super(LayerPropertiesScattering, self).get_config()
        config = {
            'n_channels': self.n_channels,
        }
        return config.update(base_config)
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.function
def propagate_layer_up (t_direct, t_diffuse, e_split_direct, e_split_diffuse, r_bottom_direct, r_bottom_diffuse, a_bottom_direct, a_bottom_diffuse):
    """
    Combines the properties of two atmospheric layers within a column: 
    usually a shallow "top layer" and a thicker "bottom layer" where this "bottom layer" spans all the 
    layers beneath the top layer including the surface. Computes the impact of multi-reflection between these layers.

    Naming conventions:
     
    The prefixes -- t, e, r, a -- correspond respectively to transmission,
    extinction, reflection, absorption.

    The suffixes "_direct" and "_diffuse" specify the type of input radiation. 
    Note, however, that an input of direct radiation may produce diffuse output,
    e.g., t_multi_direct (transmission of direct radiation through multi-reflection) 
    
    Input and Output Shape:
        Tensor with shape (n_batches, n_channels)

    Arguments:

        t_direct, t_diffuse - Direct transmission coefficients for 
            the top layer. 

        e_split_direct, e_split_diffuse - The split of extinguised  
            radiation into transmitted (diffuse), reflected,
            and absorbed components. These components sum to 1.0.
            
        r_bottom_direct, r_bottom_diffuse - The reflection 
            coefficients for bottom layer.

        a_bottom_direct, a_bottom_diffuse - The absorption coefficients
            for the bottom layer. 
            
    Returns:

        t_multi_direct, t_multi_diffuse - The transmission coefficients for 
            radiation that is multi-reflected (as opposed to directly transmitted, 
            e.g., t_direct, t_diffuse)

        r_multi_direct, r_multi_diffuse - The reflection coefficients 
            for the top layer after accounting for multi-reflection
            with the bottom layer

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

    #tf.debugging.assert_near(r_bottom_direct + a_bottom_direct, 1.0, rtol=1e-2, atol=1e-2, message="Bottom Direct", summarize=5)

    #tf.debugging.assert_near(r_bottom_diffuse + a_bottom_diffuse, 1.0, rtol=1e-2, atol=1e-2, message="Bottom Diffuse", summarize=5)

    # The top layer splits the direct beam into transmitted and extinguished components
    e_direct = 1.0 - t_direct
    
    # The top layer also splits the downward diffuse flux into transmitted and extinguished components
    e_diffuse = 1.0 - t_diffuse

    # The top layer further splits each extinguished component into transmitted, reflected,
    # and absorbed components
    e_t_direct, e_r_direct, e_a_direct = e_split_direct[:,:,0:1], e_split_direct[:,:,1:2],e_split_direct[:,:,2:]
    e_t_diffuse, e_r_diffuse, e_a_diffuse = e_split_diffuse[:,:,0:1], e_split_diffuse[:,:,1:2],e_split_diffuse[:,:,2:]

    #tf.debugging.assert_near(e_t_direct + e_r_direct + e_a_direct, 1.0, rtol=1e-3, atol=1e-3, message="Extinction Direct", summarize=5)

    #tf.debugging.assert_near(e_t_diffuse + e_r_diffuse + e_a_diffuse, 1.0, rtol=1e-3, atol=1e-3, message="Extinction Diffuse", summarize=5)

    #print(f"e_a_diffuse.shape = {e_a_diffuse.shape}")

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
    eps = 1.0e-06
    d = 1.0 / (1.0 - e_diffuse * e_r_diffuse * r_bottom_diffuse + eps)

    t_multi_direct = t_direct * r_bottom_direct * e_diffuse * e_r_diffuse * d + \
        e_direct * e_t_direct * d # good
    
    a_bottom_multi_direct = t_direct * a_bottom_direct + t_multi_direct * a_bottom_diffuse # good

    r_bottom_multi_direct = t_direct * r_bottom_direct * d + e_direct * e_t_direct * r_bottom_diffuse * d # good

    a_top_multi_direct = e_direct * e_a_direct + r_bottom_multi_direct * e_diffuse * e_a_diffuse # good

    r_multi_direct = e_direct * e_r_direct + r_bottom_multi_direct * (t_diffuse + e_diffuse * e_t_diffuse) # good

    # These should sum to 1.0
    #total_direct = a_bottom_multi_direct + a_top_multi_direct + r_multi_direct
    #tf.debugging.assert_near(total_direct, 1.0, rtol=1e-3, atol=1e-3, message="Total Direct Error", summarize=5)
    # Loss of flux should equal absorption
    #diff_flux = 1.0 - t_direct - t_multi_direct + r_bottom_multi_direct - r_multi_direct 
    #tf.debugging.assert_near(diff_flux, a_top_multi_direct, rtol=1e-3, atol=1e-3, message="Diff Flux Direct", summarize=5)

    # Multi-reflection for diffuse flux

    t_multi_diffuse = \
        t_diffuse * r_bottom_diffuse * e_diffuse * e_r_diffuse * d + \
        e_diffuse * e_t_diffuse * d  # good
    
    a_bottom_multi_diffuse = t_diffuse * a_bottom_diffuse + t_multi_diffuse * a_bottom_diffuse # good

    r_bottom_multi_diffuse = t_diffuse * r_bottom_diffuse * d + e_diffuse * e_t_diffuse * r_bottom_diffuse * d # good
    
    a_top_multi_diffuse = e_diffuse * e_a_diffuse + r_bottom_multi_diffuse * e_diffuse * e_a_diffuse # good

    r_multi_diffuse = e_diffuse * e_r_diffuse + r_bottom_multi_diffuse * (t_diffuse + e_diffuse * e_t_diffuse) # good

    #total_diffuse = a_bottom_multi_diffuse + a_top_multi_diffuse + r_multi_diffuse
    #tf.debugging.assert_near(total_diffuse, 1.0, rtol=1e-3, atol=1e-3, message="Total Diffuse Error", summarize=5)
    #diff_flux = 1.0 - t_diffuse - t_multi_diffuse + r_bottom_multi_diffuse - r_multi_diffuse
    #tf.debugging.assert_near(diff_flux, a_top_multi_diffuse, rtol=1e-3, atol=1e-3, message="Diff Flux Diffuse", summarize=5)

    #tf.debugging.assert_near(r_multi_direct + a_top_multi_direct + a_bottom_multi_direct, 1.0, rtol=1e-2, atol=1e-2, message="Top Direct", summarize=5)

    #tf.debugging.assert_near(r_multi_diffuse + a_top_multi_diffuse + a_bottom_multi_diffuse, 1.0, rtol=1e-2, atol=1e-2, message="Top Diffuse", summarize=5)

    return t_multi_direct, t_multi_diffuse, \
            r_multi_direct, r_multi_diffuse, \
            r_bottom_multi_direct, r_bottom_multi_diffuse, \
            a_top_multi_direct, a_top_multi_diffuse, \
            a_bottom_multi_direct, a_bottom_multi_diffuse

class UpwardPropagationCell(Layer):
    def __init__(self, n_channels, **kwargs):
        super().__init__(**kwargs)
        #self.state_size = ((n_channels, 1), (n_channels,1), (n_channels, 1), (n_channels, 1))
        self.state_size = [tf.TensorShape([n_channels * 4])]
        #self.state_size = [tf.TensorShape([n_channels, 1]), tf.TensorShape([n_channels, 1]), tf.TensorShape([n_channels, 1]), tf.TensorShape([n_channels, 1])]
        self.output_size = tf.TensorShape([n_channels, 8])
        self._n_channels = n_channels

    def call(self, input_at_i, states_at_i):
        #print("***")
        t_direct, t_diffuse, e_split_direct, e_split_diffuse = input_at_i[:,:,0:1], input_at_i[:,:,1:2], input_at_i[:,:,2:5],input_at_i[:,:,5:]

        #print(f"Enter upward RNN, state.len = {len(states_at_i)} and state[0].shape = {states_at_i[0].shape}")
        #print(f"t_direct  = {tf.get_static_value(t_direct)}")
        #print(f't_direct shape = {t_direct.shape}')
        #print(f'e_split_diffuse shape = {e_split_diffuse.shape}')

        #r_bottom_direct, r_bottom_diffuse, a_bottom_direct, a_bottom_diffuse = states_at_i

        reshaped_state = tf.reshape(states_at_i[0], (-1,self._n_channels,4))

        r_bottom_direct, r_bottom_diffuse, a_bottom_direct, a_bottom_diffuse = reshaped_state[:,:,0:1], reshaped_state[:,:,1:2], reshaped_state[:,:,2:3], reshaped_state[:,:,3:4]
        
        #print(f"r_bottom_direct shape = {r_bottom_direct.shape}")

        tmp = propagate_layer_up (t_direct, t_diffuse, e_split_direct, e_split_diffuse, r_bottom_direct, r_bottom_diffuse, a_bottom_direct, a_bottom_diffuse)

        #print(f"tmp = {tmp}")

        t_multi_direct, t_multi_diffuse, \
            r_multi_direct, r_multi_diffuse, \
            r_bottom_multi_direct, r_bottom_multi_diffuse, \
            a_top_multi_direct, a_top_multi_diffuse, \
            a_bottom_multi_direct, a_bottom_multi_diffuse= tmp

        output_at_i = tf.concat([t_direct, t_diffuse, t_multi_direct, t_multi_diffuse, 
                                 r_bottom_multi_direct, r_bottom_multi_diffuse, a_top_multi_direct, a_top_multi_diffuse], axis=2)
        #a_bottom_multi_direct, a_bottom_multi_diffuse], axis=2)

        #print(" ")
        #print(f"Upward Prop, output.shape = {output_at_i.shape}")
        #print(f"Upward Prop, r_multi_direct.shape = {r_multi_direct.shape}")
        
        #state_at_i_plus_1 = [r_multi_direct, r_multi_diffuse, a_top_multi_direct, a_top_multi_diffuse]

        state_at_i_plus_1 = tf.concat([r_multi_direct, r_multi_diffuse, a_top_multi_direct + a_bottom_multi_direct, a_top_multi_diffuse + a_bottom_multi_diffuse], axis=2)

        state_at_i_plus_1 = tf.reshape(state_at_i_plus_1,(-1,self._n_channels * 4))

        #print("*")
        #print(" ")
        return output_at_i, [state_at_i_plus_1]


    def get_config(self):
        base_config = super(UpwardPropagationCell, self).get_config()
        config = {
            'n_channels': self._n_channels,
        }
        return config.update(base_config)
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class DownwardPropagationCellDirect(Layer):
    def __init__(self,n_channels,**kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.state_size = [tf.TensorShape([self.n_channels, 1])]
        self.output_size = tf.TensorShape([self.n_channels, 1])

    def call(self, input_at_i, states_at_i):

        flux_down_above_direct, = states_at_i

        #print(f"RNN: flux_down_above.shape= {flux_down_above_direct.shape}")

        t_direct = input_at_i

        #print(f"RNN: t_direct.shape= {t_direct.shape}")

        # Will want this later when incorporate surface interactions
        #absorbed_flux_bottom = flux_down_above_direct * a_bottom_multi_direct + \
        #flux_down_above_diffuse * a_bottom_multi_diffuse

        flux_down_below_direct = flux_down_above_direct * t_direct
        
        output_at_i = flux_down_below_direct
         
        state_at_i_plus_1=[flux_down_below_direct,]

        #print ("Downward prop")
        return output_at_i, state_at_i_plus_1

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0],self.n_channels,1])

    def get_config(self):
        base_config = super(DownwardPropagationCellDirect, self).get_config()
        config = {
                'n_channels': self.n_channels,
        }
        return config.update(base_config)
            
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DownwardPropagationCell(Layer):
    def __init__(self,n_channels, **kwargs):
        super().__init__(**kwargs)
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

        #print(f"Downward: a_top_multi_direct shape = {a_top_multi_direct.shape}")
        #print(f"Downward: a_top_multi_diffuse shape = {a_top_multi_diffuse.shape}")

        absorbed_flux_top = flux_down_above_direct * a_top_multi_direct + \
                        flux_down_above_diffuse * a_top_multi_diffuse

        # Will want this later when incorporate surface interactions
        #absorbed_flux_bottom = flux_down_above_direct * a_bottom_multi_direct + \
        #flux_down_above_diffuse * a_bottom_multi_diffuse

        flux_down_below_direct = flux_down_above_direct * t_direct
        flux_down_below_diffuse = flux_down_above_direct * t_multi_direct + \
                                flux_down_above_diffuse * (t_diffuse + t_multi_diffuse)
        flux_up_below_diffuse = flux_down_above_direct * r_bottom_multi_direct + flux_down_above_diffuse * r_bottom_multi_diffuse
        
        output_at_i = tf.concat([flux_down_below_direct, flux_down_below_diffuse, flux_up_below_diffuse, absorbed_flux_top], axis=2) #, #absorbed_flux_bottom

        #print(f"Downward: absorbed_flux_top shape = {absorbed_flux_top.shape}")

        #print(f"Downward output shape = {output_at_i.shape}")
         
        #state_at_i_plus_1 = flux_down_below_direct, flux_down_below_diffuse
        state_at_i_plus_1=tf.concat([flux_down_below_direct, flux_down_below_diffuse], axis=2)
        state_at_i_plus_1=tf.reshape(state_at_i_plus_1,(-1,self._n_channels*2))

        return output_at_i, state_at_i_plus_1

    def get_config(self):
        base_config = super(DownwardPropagationCell, self).get_config()
        config = {
                'n_channels': self.n_channels,
        }
        return config.update(base_config)
            
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ConsolidateFluxDirect(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, input):
        #print("starting: consolidated flux")
        flux_down_above_direct, flux_down_below_direct = input

        # Add layers dimension
        flux_down_above_direct = tf.expand_dims(flux_down_above_direct, axis=1)

        flux_down_direct = tf.concat([flux_down_above_direct,flux_down_below_direct], axis=1)

        #print(f'Consolidate flux_down.shape = {flux_down_direct.shape}')

        # Sum across channels

        flux_down_direct = tf.squeeze(flux_down_direct,axis=3)     
        flux_down_direct = tf.math.reduce_sum(flux_down_direct, axis=2)


        return flux_down_direct

    def compute_output_shape(self, input_shape):
        return [tf.TensorShape([input_shape[0][0],input_shape[2][1] + 1]),]
    
    def get_config(self):
        base_config = super(ConsolidateFluxDirect, self).get_config()
        config = {
        }
        return config.update(base_config)
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ConsolidateFlux(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, input):
        #print("starting: consolidated flux")
        flux_down_above_direct, flux_down_above_diffuse, \
        flux_down_below_direct, flux_down_below_diffuse, \
        flux_up_above_diffuse, flux_up_below_diffuse, \
        absorbed_flux_top = input

        # Add layers dimension
        flux_down_above_direct = tf.expand_dims(flux_down_above_direct, axis=1)
        flux_down_above_diffuse = tf.expand_dims(flux_down_above_diffuse, axis=1)
        flux_up_above_diffuse = tf.expand_dims(flux_up_above_diffuse, axis=1)

        flux_down_direct = tf.concat([flux_down_above_direct,flux_down_below_direct], axis=1)
        flux_down_diffuse = tf.concat([flux_down_above_diffuse,flux_down_below_diffuse], axis=1)
        flux_up_diffuse = tf.concat([flux_up_above_diffuse, flux_up_below_diffuse], axis=1)

        # Sum across channels
        flux_down_direct = tf.math.reduce_sum(flux_down_direct, axis=2)
        flux_down_diffuse = tf.math.reduce_sum(flux_down_diffuse, axis=2)
        flux_up = tf.math.reduce_sum(flux_up_diffuse, axis=2)

        flux_down = flux_down_direct + flux_down_diffuse
        flux_down = tf.squeeze(flux_down,axis=2)
        flux_down_direct = tf.squeeze(flux_down_direct,axis=2)
        flux_up = tf.squeeze(flux_up,axis=2)

        #print(f"Consolidation: absorbed_flux_top = {absorbed_flux_top.shape}")

        absorbed_flux = tf.math.reduce_sum(absorbed_flux_top, axis=2)
        absorbed_flux = tf.squeeze(absorbed_flux,axis=2)

        #print(f"Consolidation: direct_down = {flux_down_direct.shape}")
        #print(f"Consolidation: direct = {flux_down.shape}")
        #print(f"Consolidation: up = {flux_up.shape}")
        #print(f"Consolidation: absorbed_flux = {absorbed_flux.shape}")

        #print("consolidated flux")
        return flux_down_direct, flux_down, flux_up, absorbed_flux

    def compute_output_shape(self, input_shape):
        return [tf.TensorShape([input_shape[0][0],input_shape[2][1] + 1]),
                tf.TensorShape([input_shape[0][0],input_shape[2][1] + 1]), 
                tf.TensorShape([input_shape[0][0],input_shape[2][1] + 1]), 
                tf.TensorShape([input_shape[0][0],input_shape[2][1]])]
    def get_config(self):
        base_config = super(ConsolidateFlux, self).get_config()
        config = {
        }
        return config.update(base_config)
    @classmethod
    def from_config(cls, config):
        return cls(**config)

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
    
    def get_config(self):
        base_config = super(VerificationLayer, self).get_config()
        config = {
        }
        return config.update(base_config)
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    


#used
def heating_rate_loss(y_true, y_pred, toa_weighting_profile, delta_pressure):
    heat_true = absorbed_flux_to_heating_rate(y_true, delta_pressure)
    heat_pred = absorbed_flux_to_heating_rate(y_pred, delta_pressure)
    error = tf.sqrt(tf.reduce_mean(tf.square(toa_weighting_profile * (heat_true - heat_pred))))
    return error

def heating_rate_loss_direct(y_true, y_pred, toa_weighting_profile, delta_pressure):
    absorbed_true = y_true[:,:-1] - y_true[:,1:]
    absorbed_pred = y_pred[:,:-1] - y_pred[:,1:]
    error = heating_rate_loss(absorbed_true, absorbed_pred, toa_weighting_profile, delta_pressure)
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
def ukkonen_loss_direct(y_true, y_pred, weight_profile, toa_weighting_profile, delta_pressure):
    flux_loss = weighted_loss(y_true, y_pred, weight_profile)
    hr_loss = heating_rate_loss_direct(y_true, y_pred, toa_weighting_profile, delta_pressure)
    alpha   = 1.0e-4
    return alpha * hr_loss + (1.0 - alpha) * flux_loss

def ukkonen_loss(y_true, y_pred, target_absorbed_flux, absorbed_flux,weight_profile, toa_weighting_profile, delta_pressure):
    flux_loss = weighted_loss(y_true, y_pred, weight_profile)
    hr_loss = heating_rate_loss(target_absorbed_flux, absorbed_flux, toa_weighting_profile, delta_pressure)
    alpha   = 1.0e-4
    return alpha * hr_loss + (1.0 - alpha) * flux_loss

def henry_loss(y_true, y_pred, target_flux_down_direct, flux_down_direct, target_absorbed_flux, absorbed_flux,weight_profile, toa_weighting_profile, delta_pressure):
    flux_loss = weighted_loss(y_true, y_pred, weight_profile)
    hr_loss = heating_rate_loss(target_absorbed_flux, absorbed_flux, toa_weighting_profile, delta_pressure)
    hr_direct_loss = heating_rate_loss_direct(target_flux_down_direct, flux_down_direct, toa_weighting_profile, delta_pressure)
    alpha   = 1.0e-4
    return alpha * (hr_loss + hr_direct_loss) + (1.0 - alpha) * flux_loss

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
    factor_1 = 0.92  # decrease in original positive weights #0.92, 1.1, 0.86; 0.9, 1.1
    factor_2 = 0.1  # Initial fraction of possible weight for negative weights #0.1, 0.2, 0.35; 0.3, 0.2
    ke_index_list = [0, 1, 2, 3, 20, 29, 38, 47, 56]
    for layer in model.layers:
        if layer.name == 'optical_depth':
            layer_weights = layer.get_weights()
            new_weights = []
            for k, weights in enumerate(layer_weights):
                positive_weights = [x for x in np.nditer(weights) if x > 0.0]
                n_positive = len(positive_weights)
                n_negative = weights.size - n_positive
                if n_negative == 0 or k not in ke_index_list:
                    new_weights.append(weights)
                elif n_positive == 0:
                    new_weights.append(np.full(shape=weights.shape, fill_value=2.0e-02))
                else:
                    new_positive_weight = factor_2 * sum(positive_weights) / n_positive
                    modified_weights = weights * factor_1
                    modified_weights = [x if x > 0.0 else new_positive_weight for x in np.nditer(modified_weights)]
                    np_modified_weights = np.reshape(np.array(modified_weights), weights.shape)
                    new_weights.append(np_modified_weights)
            layer.set_weights(new_weights)
    return model


def modify_weights_2(model):
    for layer in model.layers:
        if layer.name == 'flux_down_above_direct':
            layer_weights = layer.get_weights()
            new_weights = []
            for k, weights in enumerate(layer_weights):
                if k > 0:
                    new_weights.append(weights)
                else:
                    modified_weights = [np.sqrt(x) if x > 0.0 else -np.sqrt(-x) for x in np.nditer(weights)]
                    modified_weights = np.reshape(np.array(modified_weights), weights.shape)
                    new_weights.append(modified_weights)
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
    l2_regularization = 0.00001

    datadir     = "/home/hws/tmp/"
    filename_training       = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"
    filename_validation   = datadir + "/RADSCHEME_data_g224_CAMS_2014.2.nc"
    filename_testing  = datadir +  "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"
    log_dir = datadir + "/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename_direct_model = datadir + "/Direct_Model-"
    filename_full_model = datadir + "/Full_Model-"
    model_name = "mass.2.h2o_sq." #"2.Dropout."
    use_direct_model = False

    # Computing optical depth for each layer

    t_p = Input(shape=(n_layers, 3), batch_size=batch_size, name="t_p_input")

    lw = Input(shape=(n_layers, 2), batch_size=batch_size, name="lw_input")

    h2o = Input(shape=(n_layers, 1), batch_size=batch_size, name="h2o_input") 

    o3 = Input(shape=(n_layers, 1), batch_size=batch_size, name="o3_input") 

    co2 = Input(shape=(n_layers, 1), batch_size=batch_size, name="co2_input") 

    o2 = Input(shape=(n_layers, 1), batch_size=batch_size, name="o2_input") 

    u = Input(shape=(n_layers, 1), batch_size=batch_size, name="u_input") 

    n2o = Input(shape=(n_layers, 1), batch_size=batch_size, name="n2o_input") 

    ch4 = Input(shape=(n_layers, 1), batch_size=batch_size, name="ch4_input") 

    h2o_sq = Input(shape=(n_layers, 1), batch_size=batch_size, name="h2o_sq_input") 

    tau = TimeDistributed(OpticalDepth(n_channels), name="optical_depth")([lw, h2o, o3, co2, u, n2o, ch4, h2o_sq, t_p])

    # If only running direct transmission model, compute direct transmission 
    # coefficient for each layer

    mu = Input(shape=(n_layers, 1), batch_size=batch_size, name="mu_input") 

    t_direct = TimeDistributed(LayerPropertiesDirect(), name="layer_properties")([tau, mu, u])

    # Initializing downwelling radiative flux and splitting it into "channels"

    total_flux_down_above_direct = Input(shape=(1), batch_size=batch_size, name="flux_down_above_direct_input")

    flux_down_above_direct = Dense(units=n_channels,
                                   activation=None, #'softmax', # softmax is done below!
                                   use_bias=False,
                                    #kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
                                   #kernel_initializer=initializers.RandomUniform(minval=0.1, maxval=1.0),
                                   #kernel_constraint=tf.keras.constraints.NonNeg(), 
                                    kernel_initializer=initializers.RandomUniform(minval=0.4, maxval=0.6),name='flux_down_above_direct')(total_flux_down_above_direct)
    
    flux_down_above_direct = tf.keras.layers.Dropout(0.0)(flux_down_above_direct) # 0.15, 0.075, # 0.0375
    flux_down_above_direct = tf.nn.softmax(flux_down_above_direct)

    flux_down_above_direct = tf.expand_dims(flux_down_above_direct, axis=2) # Need to add additional dimension since each channel is independent.

    # Downward propagation of direct radiative flux

    flux_down_below_direct = RNN(DownwardPropagationCellDirect(n_channels), return_sequences=True, return_state=False, go_backwards=False, time_major=False)(inputs=t_direct, initial_state=[flux_down_above_direct,])

    flux_down_direct = ConsolidateFluxDirect()((flux_down_above_direct, flux_down_below_direct))

    toa = Input(shape=(1), batch_size=batch_size, name="toa_input")
    target_flux_down_direct = Input(shape=(n_levels), batch_size=batch_size, name="target_flux_down_direct_input")

    delta_pressure = Input(shape=(n_layers), batch_size=batch_size, name="delta_pressure_input")

    weight_profile_direct = 1.0 / tf.reduce_mean(target_flux_down_direct, axis=0, keepdims=True)

    if use_direct_model:
        direct_model = Model(inputs=[mu, lw, h2o, o3, co2, o2, u, n2o, ch4, h2o_sq, t_p, total_flux_down_above_direct, toa, target_flux_down_direct, delta_pressure], 
        outputs=[flux_down_direct])

        direct_model.add_metric(heating_rate_loss_direct(target_flux_down_direct, flux_down_direct, toa, delta_pressure),name="hr")

        direct_model.add_metric(flux_rmse(target_flux_down_direct, flux_down_direct, toa),name="flux_rmse")
        
        direct_model.add_loss(ukkonen_loss_direct(target_flux_down_direct, flux_down_direct, weight_profile_direct,toa, delta_pressure)) #,name="ukk_loss")

        direct_model.compile(
            optimizer=optimizers.Adam(),
            loss_weights= [1.0],
            metrics=[[OriginalLoss(1400)]],
        )
        direct_model.summary()

    if False:
        n_epochs =1100
        direct_model.load_weights((filename_direct_model + model_name + str(n_epochs)))
        for layer in direct_model.layers:
            if layer.name == 'flux_down_above_direct':
                print(f'flux_down_above_direct.weights = {layer.weights}')
            if layer.name == 'optical_depth':
                print("Optical Depth layers")
                for k, weights in enumerate(layer.weights):
                    print(f'Weights {k}: {weights}')

        direct_model = modify_weights_1(direct_model)
        #writer = tf.summary.create_file_writer(log_dir)
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=False, profile_batch=(2, 4))

    if use_direct_model:
        training_inputs, training_outputs = load_data_direct(filename_training, n_channels)
        validation_inputs, validation_outputs = load_data_direct(filename_validation, n_channels)
        while n_epochs < epochs:
            history = direct_model.fit(x=training_inputs, y=training_outputs,
                    epochs = epochs_period, batch_size=batch_size,
                    shuffle=True, verbose=1,
                    validation_data=(validation_inputs, validation_outputs),
                    callbacks = []) #[tensorboard_callback])
                    
            n_epochs = n_epochs + epochs_period

            for layer in direct_model.layers:
                if layer.name == 'flux_down_above_direct':
                    print(f'flux_down_above_direct.weights = {layer.weights}')
                if layer.name == 'optical_depth':
                    print("")
                    for k, weights in enumerate(layer.weights):
                        print(f'Weights {k}: {weights}')


            print(f"Writing model weights {n_epochs}")
            direct_model.save_weights(filename_direct_model + model_name + str(n_epochs)) 

    # Connect graph for full model

    # Learn coefficient for diffuse mu (effective cosine of zenith angle)
    if not use_direct_model:
        mu_bar_input = Input(shape=(1), batch_size=batch_size, name="mu_bar_input") 
        mu_bar = Dense(units=1,kernel_initializer='zeros', use_bias=False, activation="sigmoid",name="mu_bar")(mu_bar_input)
        mu_bar = tf.repeat(tf.expand_dims(mu_bar,axis=1),repeats=n_layers,axis=1)

        # Compute properties of each layer:
        #   Direct and diffuse transmission coefficients
        #   Split of extinguished radiation into transmitted, 
        #         reflected, absorbed components

        layer_properties = TimeDistributed(LayerPropertiesScattering(n_channels), name="layer_properties")([tau, mu, mu_bar, lw, h2o, o3, co2, u, n2o, ch4])

        # Compute multireflection among layers. For each layer, resolve into
        # absorption (a) and reflection (r) where these describe 
        # the cumulative effect for downward radiation on the current layer 
        # and all the layers beneath it including the surface
        # Note: computes two sets of coefficients corresponding
        # to direct and diffuse radiation

        surface_albedo_direct = Input(shape=(n_channels, 1), batch_size=batch_size, name="surface_albedo_direct_input")

        surface_albedo_diffuse = Input(shape=(n_channels, 1), batch_size=batch_size, name="surface_albedo_diffuse_input")

        surface_absorption_direct = Input(shape=(n_channels, 1), batch_size=batch_size, name="surface_absorption_direct_input")

        surface_absorption_diffuse = Input(shape=(n_channels, 1), batch_size=batch_size, name="surface_absorption_diffuse_input")

        initial_state=tf.concat([surface_albedo_direct, surface_albedo_diffuse,
                    surface_absorption_direct, surface_absorption_diffuse], axis=2)
        
        initial_state = tf.reshape(initial_state,(-1, n_channels * 4))

        #t_direct_1 = tf.identity(layer_properties[:,0,0:3,0:1])

        multireflection_layer_parameters, upward_state = RNN(UpwardPropagationCell(n_channels), return_sequences=True, return_state=True, go_backwards=True, time_major=False)(inputs=layer_properties, initial_state = initial_state)

        #t_direct_2 = tf.identity(layer_properties[:,0,0:3,0:1])

        upward_state = tf.reshape(upward_state,(-1,n_channels,4))
        r_multi_direct = upward_state[:,:,0:1]  # reflection at top of atmosphere

        # Downward propagation of full model including direct and diffuse flux
        flux_down_above_diffuse = Input(shape=(n_channels, 1), batch_size=batch_size, name="flux_down_above_diffuse_input") # initialized to zero

        flux_up_above_diffuse = tf.multiply(flux_down_above_direct,r_multi_direct)

        initial_state_down=tf.concat([flux_down_above_direct, flux_down_above_diffuse], axis=2)

        initial_state_down=tf.reshape(initial_state_down,(-1,n_channels*2))

        multireflection_layer_parameters = tf.reverse(multireflection_layer_parameters, axis=[1])

        """
        multireflection_layer_parameters = tf.concat([tf.reverse(layer_properties[:,:,:,0:1], axis=[1]), tf.reverse(layer_properties[:,:,:,1:2], axis=[1]),tmp_tensor],axis=3)
        """
        # Downward propagation for flux
        # Computes transmitted and reflected flux and absorbed flux at each layer
        output = RNN(DownwardPropagationCell(n_channels), return_sequences=True, return_state=False, go_backwards=False, time_major=False)(inputs=multireflection_layer_parameters, initial_state=initial_state_down)

        #print(f"downward output shape = {output.shape}")

        flux_down_below_direct, flux_down_below_diffuse, \
        flux_up_below_diffuse, absorbed_flux_top = output[:,:,:,0:1],output[:,:,:,1:2],output[:,:,:,2:3],output[:,:,:,3:4]

        flux_inputs = (flux_down_above_direct, flux_down_above_diffuse,
            flux_down_below_direct, flux_down_below_diffuse, 
            flux_up_above_diffuse, flux_up_below_diffuse,
            absorbed_flux_top)

        #model_error = VerificationLayer()([flux_down_above_direct, layer_properties[:,:,:,0:1], flux_down_below_direct])

        ##print(f"model error = {model_error}")
            
        flux_down_direct, flux_down, flux_up, absorbed_flux = ConsolidateFlux()(flux_inputs)


        # This is just to force the VerificationLayer to run
        #heating_rate = heating_rate + tf.expand_dims(model_error, axis=1)

        target_flux_down = Input(shape=(n_levels), batch_size=batch_size, name="target_flux_down_input")

        target_flux_up = Input(shape=(n_levels), batch_size=batch_size, name="target_flux_up_input")
        
        target_absorbed_flux = Input(shape=(n_layers), batch_size=batch_size, name="target_absorbed_flux_input")

        full_model = Model(inputs=[mu, mu_bar_input, lw, h2o, o3, co2, o2, u, n2o, ch4, h2o_sq,t_p, 
                                surface_albedo_direct, surface_albedo_diffuse,
                                surface_absorption_direct, surface_absorption_diffuse,
                                total_flux_down_above_direct,
                                    flux_down_above_diffuse, toa, target_flux_down_direct,
                                    target_flux_down,
                                    target_flux_up,
                                    target_absorbed_flux, delta_pressure], 
        outputs=[flux_down_direct, flux_down, flux_up, absorbed_flux])


        full_model.add_metric(heating_rate_loss(target_absorbed_flux, absorbed_flux, toa, delta_pressure),name="hr")

        full_model.add_metric(heating_rate_loss_direct(target_flux_down_direct, flux_down_direct, toa, delta_pressure),name="hr_direct")

        predicted_flux = tf.concat([flux_down, flux_up], axis=-1)
        target_flux  = tf.concat([target_flux_down, target_flux_up], axis=-1)

        weight_profile_full = 1.0 / tf.reduce_mean(target_flux, axis=0, keepdims=True)

        full_model.add_metric(flux_rmse(target_flux, predicted_flux, toa),name="flux_rmse")
        
        #full_model.add_loss(ukkonen_loss(target_flux, predicted_flux, target_absorbed_flux, absorbed_flux, weight_profile_full, toa, delta_pressure))

        full_model.add_loss(henry_loss(target_flux, predicted_flux, target_flux_down_direct, flux_down_direct, target_absorbed_flux, absorbed_flux, weight_profile_full, toa, delta_pressure))

        full_model.compile(
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
            #metrics=[[OriginalLoss(1400)]],
        #{flux_down.name:'mse', flux_up.name : 'mse', heating_rate.name: 'mse'},
        )
        full_model.summary()

        print(f"full_model.metrics_names = {full_model.metrics_names}")

        
        #output = model(inputs=validation_inputs)

        #print(f"len of output = {len(output)}")


        if True:
            n_epochs_full = 500
            n_epochs = n_epochs_full
            full_model.load_weights((filename_full_model + model_name + str(n_epochs_full)))

        if False:
            n_epochs_full = 7
            n_epochs = n_epochs_full
            n_epochs_direct = 75
            full_model.load_weights((filename_full_model + model_name + str(n_epochs_full)))

            direct_model.load_weights((filename_direct_model + model_name + str(n_epochs_direct)))

            for layer in full_model.layers:
                if layer.name == 'flux_down_above_direct':
                    for direct_layer in direct_model.layers:
                        if direct_layer.name == layer.name:
                            layer.set_weights(direct_layer.get_weights())
                            layer.trainable = False

                if layer.name == 'optical_depth':
                    for direct_layer in direct_model.layers:
                        if direct_layer.name == layer.name:
                            layer.set_weights(direct_layer.get_weights())
                            layer.trainable = False

            for layer in full_model.layers:
                if layer.name == 'flux_down_above_direct':
                    print(f'flux_down_above_direct.weights = {layer.weights}')
                if layer.name == 'optical_depth':
                    print("Optical Depth layers")
                    for k, weights in enumerate(layer.weights):
                        print(f'Weights {k}: {weights}')
            # Need to call compile 2nd time for change to 'trainable' to take effect
            full_model.compile(
                optimizer=optimizers.Adam(),
                loss_weights= [1.0],
            )

            full_model.summary()


        if False:
            n_epochs = 272
            full_model.load_weights((filename_full_model + model_name + str(n_epochs)))
            for layer in full_model.layers:
                if layer.name == 'flux_down_above_direct':
                    print(f'flux_down_above_direct.weights = {layer.weights}')
                if layer.name == 'optical_depth':
                    print("Optical Depth layers")
                    for k, weights in enumerate(layer.weights):
                        print(f'Weights {k}: {weights}')

            full_model = modify_weights_1(full_model)

            for layer in full_model.layers:
                if layer.name == 'flux_down_above_direct':
                    print(f'flux_down_above_direct.weights = {layer.weights}')
                if layer.name == 'optical_depth':
                    print("Optical Depth layers")
                    for k, weights in enumerate(layer.weights):
                        print(f'Weights {k}: {weights}')

            #writer = tf.summary.create_file_writer(log_dir)
            #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=False) # profile_batch=('2,4'))

        testing_inputs, testing_outputs = load_data_full(filename_testing, n_channels, n_coarse_code=0)


        #print("Evaluating model:")
        #full_model.evaluate(testing_inputs, testing_outputs, batch_size=batch_size)

        if not use_direct_model:
            training_inputs, training_outputs = load_data_full(filename_training, n_channels,n_coarse_code=0)
            validation_inputs, validation_outputs = load_data_full(filename_validation, n_channels,n_coarse_code=0)
            while n_epochs < epochs:
                history = full_model.fit(x=training_inputs, y=training_outputs,
                        epochs = epochs_period, batch_size=batch_size,
                        shuffle=True, verbose=1,
                        validation_data=(validation_inputs, validation_outputs))
                        #callbacks = [tensorboard_callback])
                        
                #,callbacks = [EarlyStopping(monitor='heating_rate',  patience=patience, verbose=1, \
                #                  mode='min',restore_best_weights=True),])
                
                print (" ")

                print (" ")
                n_epochs = n_epochs + epochs_period


                print("Evaluating model:")
                full_model.evaluate(testing_inputs, testing_outputs, batch_size=batch_size)
                if False:
                    for layer in full_model.layers:
                        if layer.name == 'flux_down_above_direct':
                            print(f'flux_down_above_direct.weights = {layer.weights}')
                        if layer.name == 'optical_depth':
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
                full_model.save_weights(filename_full_model + model_name + str(n_epochs)) #, save_traces=True)
                    #model = modify_weights_1(model)
                    #del model

                    #model.load_weights((filename_full_model + 'TEMP.4.' + str(n_epochs))) 


                
        """ model = tf.keras.models.load_model(filename_full_model + 'TEMP.' + str(n_epochs),
                                                custom_objects={'OpticalDepth': OpticalDepth,
                                                                'LayerProperties': LayerProperties,
                                                                'UpwardPropagationCell' : UpwardPropagationCell,
                                                                'DownwardPropagationCell' : DownwardPropagationCell,
                                                                'DenseFFN' : DenseFFN,
                                                                }) """

if __name__ == "__main__":
    train()