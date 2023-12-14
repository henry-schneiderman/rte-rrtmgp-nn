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

from RT_data_hws import absorbed_flux_to_heating_rate, load_data_2

class DenseFFN(Layer):
    """
    n_hidden[n_layers]: array of the number of nodes per layer
    Last layer has RELU activation insuring non-negative output
    """
    def __init__(self, n_hidden, n_outputs, minval, maxval, name=None, **kwargs):
        super().__init__(**kwargs)
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.minval = minval
        self.maxval = maxval
        self.hidden = [Dense(units=n, activation='elu',
                             #kernel_initializer=initializers.RandomUniform(minval=minval, maxval=maxval), 
                             kernel_initializer=initializers.GlorotUniform(),
                             bias_initializer=initializers.RandomNormal(), name=name + str(n)) for n in n_hidden]
        # RELU / softplus insures that absorption coefficient is non-negative
        self.out = Dense(units=n_outputs, activation='softplus',kernel_initializer=initializers.RandomUniform(minval=minval, maxval=maxval), name=name + 'output') 

    def call(self, X):

        for hidden in self.hidden:
            X = hidden(X)
        return self.out(X)


    def get_config(self):
        base_config = super(DenseFFN, self).get_config()
        config = {
            'n_hidden': self.n_hidden,
            'n_outputs': self.n_outputs,
            'minval' : self.minval,
            'maxval' : self.maxval,
        }
        return config.update(base_config)

    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DenseFFN_2(Layer):
    """
    n_hidden[n_layers]: array of the number of nodes per layer
    Last layer has RELU activation insuring non-negative output
    """
    def __init__(self, n_hidden, n_outputs, **kwargs):
        super().__init__(**kwargs)
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.l0 = Dense(n_hidden[0], activation=None, kernel_initializer=tf.keras.initializers.glorot_uniform(), name="layer1")
        self.bn0 = BatchNormalization()
        self.ba0 = tf.keras.layers.Activation('relu')
        self.l1 = Dense(n_hidden[1], activation=None, kernel_initializer=tf.keras.initializers.glorot_uniform(), name="layer2")
        self.bn1 = BatchNormalization()
        self.ba1 = tf.keras.layers.Activation('relu')
        self.out = Dense(n_outputs, name="output_layer", activation='softplus')

    def call(self, X):
        X = self.l0(X)
        X = self.bn0(X)
        X = self.ba0(X)
        X = self.l1(X)
        X = self.bn1(X)
        X = self.ba1(X)
        return self.out(X)

    def get_config(self):
        base_config = super(DenseFFN_2, self).get_config()
        config = {'n_hidden': self.n_hidden,
                'n_outputs': self.n_outputs,
        }
        return config.update(base_config)
            
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class OpticalDepth(Layer):
    def __init__(self, n_hidden, n_channels, is_doubled=False, **kwargs):
        super().__init__(**kwargs)

        self.n_channels = n_channels
        if is_doubled:
            self.n_half_channels = n_channels / 2
        else:
            self.n_half_channels = n_channels 
        self.n_hidden = n_hidden

        # h2o, o3, co2, n2o, ch4, uniform
        n_ke = np.array([29,13,9,3,9,13])

        self.ke_gas_net_1 = []
        self.ke_gas_net_2 = []

        # Represents a function of temperature and pressure
        # used to build a gas absorption coefficient, ke 

        for i, n in enumerate(n_ke):
            self.ke_gas_net_1.append([DenseFFN(self.n_hidden,1,minval=-1.0,maxval=1.0, name='ke_gas.' + str(i) + f".{j}.") for j in np.arange(n)])

            if is_doubled:
                self.ke_gas_net_2.append([DenseFFN(self.n_hidden,1,minval=-1.0,maxval=1.0) for _ in np.arange(n)])


        self.ke_lw_net_1 = [Dense(units=1,bias_initializer=initializers.RandomUniform(minval=0.0, maxval=1.0), activation='softplus', name='ke_lw.' + f'{i}.') for i in np.arange(self.n_half_channels)]
        self.ke_iw_net_1 = [Dense(units=1, bias_initializer=initializers.RandomUniform(minval=0.0, maxval=1.0), activation='softplus', name='ke_iw.' + f'{i}.') for i in np.arange(self.n_half_channels)]

        if is_doubled:
            self.ke_lw_net_2 = [Dense(units=1,bias_initializer=initializers.RandomUniform(minval=0.0, maxval=1.0), activation='softplus',) for _ in np.arange(self.n_half_channels)]
            self.ke_iw_net_2 = [Dense(units=1, bias_initializer=initializers.RandomUniform(minval=0.0, maxval=1.0), activation='softplus') for _ in np.arange(self.n_half_channels)]
    # Note Ukkonen does not include nitrogen dioxide (no2) in simulation that generated data
    def subcall(self, input, gas_net, lw_net, iw_net):

        t_p, composition, null_lw, null_iw = input

        # Generate multiple optical depths for each gas

        print(f"shape of composition = {composition.shape}")

        tau_gas = []
        for i, ke_gas_net in enumerate(gas_net):
            # Extinction coefficient determined by network
            ke = [net(t_p) for net in ke_gas_net]
            # Tau = ke * mass_path_for_gas
            ke = tf.convert_to_tensor(ke)
            print(f"shape of ke = {ke.shape}")
            tau_gas.append(tf.multiply(ke,tf.reshape(composition[:,i],(1,-1, 1))))

        h2o, o3, co2, n2o, ch4, u = tau_gas

        # Optical depth for each channel
        # using various combinations of gases' optical depths

        #tau_gases = tf.Variable(initial_value=np.zeros((self.n_channels, h2o.shape[1])))

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
        tau_lw = [net(null_lw) for net in lw_net]
        tau_lw = tf.convert_to_tensor(tau_lw)
        tau_lw = tf.multiply(tau_lw,tf.reshape(composition[:,6], (1,-1, 1)))

        tau_iw = [net(null_iw) for net in iw_net]
        tau_iw = tf.convert_to_tensor(tau_iw)
        tau_iw = tf.multiply(tau_iw,tf.reshape(composition[:,7], (1,-1, 1)))

        tau_gases = tf.transpose(tau_gases, perm=[1,0,2])
        tau_lw = tf.transpose(tau_lw, perm=[1,0,2])
        tau_iw = tf.transpose(tau_iw, perm=[1,0,2])

        return [tau_gases, tau_lw, tau_iw]
    
    def call(self, input):

        t_p, composition, null_lw, null_iw = input

        output_1 = self.subcall(input, self.ke_gas_net_1, self.ke_lw_net_1, self.ke_iw_net_1)

        if False:
            output_2 = self.subcall(input, self.ke_gas_net_2, self.ke_lw_net_2, self.ke_iw_net_2)

            tau_gases_1, tau_lw_1, tau_iw_1 = output_1
            tau_gases_2, tau_lw_2, tau_iw_2 = output_2

            tau_gases = tf.concat((tau_gases_1, tau_gases_2), axis=1)
            tau_lw = tf.concat((tau_lw_1, tau_lw_2), axis=1)
            tau_iw = tf.concat((tau_iw_1, tau_iw_2), axis=1)

            return [tau_gases, tau_lw, tau_iw]
        else:
            return output_1
    
    def compute_output_shape(self, input_shape):
        return [tf.TensorShape([input_shape[0][0],self.n_channels,1]), tf.TensorShape([input_shape[0][0],self.n_channels,1]), tf.TensorShape([input_shape[0][0],self.n_channels,1])]

    def get_config(self):
        base_config = super(OpticalDepth, self).get_config()
        config = {'n_channels': self.n_channels,
                    'n_hidden' : self.n_hidden,
        }
        return config.update(base_config)
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class LayerProperties(Layer):
    def __init__(self, n_hidden, n_channels, **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.n_hidden = n_hidden
        self.extinction_net = [DenseFFN(self.n_hidden,3,minval=-1.0,maxval=1.0, name=f"extinction.{i}.") for i in np.arange(self.n_channels)]
        #self.extinction_net = [DenseFFN_2(n_hidden,3) for _ in np.arange(self.n_channels)]

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

        t_direct = tf.math.exp(-tau_total / (tf.expand_dims(mu,axis=2) + 0.00001))

        print(f'Shape of t_direct = {t_direct.shape}')
        print(" ")

        # To avoid division by zero
        t_diffuse = tf.math.exp(-tau_total / (tf.expand_dims(mu_bar,axis=2) + 0.001))

        e_split_direct = tf.transpose(e_split_direct,perm=[1,0,2])
        e_split_diffuse = tf.transpose(e_split_diffuse,perm=[1,0,2])

        print(f'Shape of e_split_diffuse = {e_split_diffuse.shape}')
        print(" ")

        layer_properties = tf.concat([t_direct, t_diffuse, e_split_direct, e_split_diffuse], axis=2)

        return layer_properties

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0][0],self.n_channels,8])

    def get_config(self):
        base_config = super(LayerProperties, self).get_config()
        config = {
                'n_channels': self.n_channels,
                'n_hidden': self.n_hidden
        }
        return config.update(base_config)
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

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
    eps = 1.0e-06
    d = 1.0 / (1.0 - e_diffuse * e_r_diffuse * r_bottom_diffuse + eps)

    t_multi_direct = t_direct * r_bottom_direct * e_diffuse * e_r_diffuse * d +  e_direct * e_t_direct * d
    
    a_bottom_multi_direct = t_direct * a_bottom_direct + t_multi_direct * a_bottom_diffuse

    r_bottom_multi_direct = t_direct * r_bottom_direct * d + e_direct * e_t_direct * r_bottom_diffuse * d

    a_top_multi_direct = e_direct * e_a_direct + r_bottom_multi_direct * e_diffuse * e_a_diffuse

    r_multi_direct = e_direct * e_r_direct + r_bottom_multi_direct * (t_diffuse + e_diffuse * e_t_diffuse)

    # These should sum to 1.0
    total_direct = a_bottom_multi_direct + a_top_multi_direct + r_multi_direct
    #print(f"total_direct.shape = {total_direct.shape}")
    print(f"total_direct (should equal 1.0) = {total_direct}")
    print(f"t_direct:  = {t_direct}")
    print(f"t_diffuse:  = {t_diffuse}")
    print(f"e_direct:  = {e_direct}")
    print(f"Direct_sum (should equal 1.0) =  {e_t_direct + e_r_direct + e_a_direct}")
    print(f"Diffuse_sum (should equal 1.0) =  {e_t_diffuse + e_r_diffuse + e_a_diffuse}")
    print(f"Direct Bottom Sum:  = {r_bottom_direct + a_bottom_direct}")
    print(f"Diffuse Bottom Sum:  = {r_bottom_diffuse + a_bottom_diffuse}")

    #print(f"denominator term (should *NOT* equal 1.0) = {e_diffuse * e_r_diffuse * r_bottom_diffuse}")
    #assert isclose(total_direct, 1.0, abs_tol=1e-5)
    # Loss of flux should equal absorption
    diff_flux = 1.0 - t_direct - t_multi_direct + r_bottom_multi_direct - r_multi_direct 
    print(f"direct diff flux - abs (should equal 0.0) = {tf.get_static_value(diff_flux - a_top_multi_direct)}")

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
    print(f"total_diffuse (should equal 1.0) = {tf.get_static_value(total_diffuse)}")
    diff_flux = 1.0 - t_diffuse - t_multi_diffuse + r_bottom_multi_diffuse - r_multi_diffuse
    print(f"diffuse diff flux - abs (should equal 0.0) = {tf.get_static_value(diff_flux - a_top_multi_diffuse)}")
    #assert isclose(diff_flux, a_top_multi_diffuse, abs_tol=1e-5)

    return t_multi_direct, t_multi_diffuse, \
            r_multi_direct, r_multi_diffuse, \
            r_bottom_multi_direct, r_bottom_multi_diffuse, \
            a_top_multi_direct, a_top_multi_diffuse, \
            a_bottom_multi_direct, a_bottom_multi_diffuse

class UpwardPropagationCell(Layer):
    def __init__(self, n_channels, **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.state_size = [tf.TensorShape([self.n_channels, 1]), tf.TensorShape([self.n_channels, 1]), tf.TensorShape([self.n_channels, 1]), tf.TensorShape([self.n_channels, 1])]
        self.output_size = tf.TensorShape([self.n_channels, 10])


    def call(self, input_at_i, states_at_i):
        print("***")
        t_direct, t_diffuse, e_split_direct, e_split_diffuse = input_at_i[:,:,0:1], input_at_i[:,:,1:2], input_at_i[:,:,2:5],input_at_i[:,:,5:]

        print(f"Enter upward RNN, state.len = {len(states_at_i)} and state[0].shape = {states_at_i[0].shape}")
        print(f"t_direct  = {tf.get_static_value(t_direct)}")

        r_bottom_direct, r_bottom_diffuse, a_bottom_direct, a_bottom_diffuse = states_at_i
        
        print(f"r_bottom_direct shape = {r_bottom_direct.shape}")

        tmp = propagate_layer_up (t_direct, t_diffuse, e_split_direct, e_split_diffuse, r_bottom_direct, r_bottom_diffuse, a_bottom_direct, a_bottom_diffuse)

        t_multi_direct, t_multi_diffuse, \
            r_multi_direct, r_multi_diffuse, \
            r_bottom_multi_direct, r_bottom_multi_diffuse, \
            a_top_multi_direct, a_top_multi_diffuse, \
            a_bottom_multi_direct, a_bottom_multi_diffuse= tmp

        output_at_i = tf.concat([t_multi_direct, t_multi_diffuse, 
                                r_multi_direct, r_multi_diffuse, 
                                 r_bottom_multi_direct, r_bottom_multi_diffuse,
        a_top_multi_direct, a_top_multi_diffuse,  
        a_bottom_multi_direct, a_bottom_multi_diffuse], axis=2)

        print(f"Upward Prop, r_multi_direct.shape = {r_multi_direct.shape}")
        
        state_at_i_plus_1 = [r_multi_direct, r_multi_diffuse, a_top_multi_direct + a_bottom_multi_direct, a_top_multi_diffuse + a_bottom_multi_diffuse]

        print("*")

        print(" ")
        return output_at_i, state_at_i_plus_1

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0],self.n_channels,10])

    def get_config(self):
        base_config = super(UpwardPropagationCell, self).get_config()
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

        i = input_at_i

        t_direct = i[:,:,0:1]

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
    n_channels = 29 #58
    batch_size  = 2048
    epochs      = 100000
    n_epochs    = 0
    epochs_period = 20
    patience    = 1000 #25

    datadir     = "/data-T1/hws/tmp/"
    filename_training       = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"
    filename_validation   = datadir + "/RADSCHEME_data_g224_CAMS_2014.2.nc"
    filename_testing  = datadir +  "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"
    log_dir = datadir + "/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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

    null_toa_input = Input(shape=(0), batch_size=batch_size, name="null_toa_input")

    #flux_down_above_direct = Dense(units=n_channels,bias_initializer='ones', activation='softmax')(null_toa_input)

    flux_down_above_direct = Dense(units=n_channels,bias_initializer=initializers.RandomUniform(minval=0.1, maxval=1.0), activation='softmax', name='null_toa_dense')(null_toa_input)

    #flux_down_above_direct = tf.keras.layers.Dropout(0.2)(flux_down_above_direct)

    print(f"flux_down_above_direct.shape={flux_down_above_direct.shape}")

    flux_down_above_direct = tf.expand_dims(flux_down_above_direct,2)

    initial_state_down=[flux_down_above_direct,]

    downward_input = layer_properties

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
                callbacks = []) #[tensorboard_callback])
                
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