import os

import numpy as np


import tensorflow as tf
from tensorflow.keras import losses, optimizers, layers, Input, Model, Layer, Sequential
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Dense,TimeDistributed


class NoInputLayer(Model):
    def __init__(self,num_outputs,minval=0.0,maxval=1.0):
        super().__init__()
        self.w = tf.random.normal((num_outputs),minval=minval,maxval=maxval)
        self.w = tf.Variable(self.w)
    def call(self):
        return(self.w)

class DenseFFN(Model):
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


class AtmLayer(Model):
    def __init__(self, n_hidden_gas, n_hidden_ext):
        super().__init__()

        self._n_nets=27

        n_gas = {}
        n_gas['h2o']=21
        n_gas['h2o_squared']=27
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
    def call(self, temp, pressure, composition, lwp, iwp, mu):
        X = tf.concat([temp,pressure],axis=-1)

        k = {}
        for gas, net in self.ke_gas.items():
            k[gas] = [n(X) for n in net]
            k[gas] = tf.mult(k[gas],composition[gas])

        tau_lw = [net() for net in self.ke_lw]
        tau_lw = tf.mult(tau_lw,lwp)
        tau_iw = [net() for net in self.ke_iw]
        tau_iw = tf.mult(tau_iw,iwp)

        h2o = k['h2o']
        h2o_sq = k['h2o_sq']
        o3 = k['o3']
        co2 = k['co2']
        n2o = k['n2o']
        ch4 = k['ch4']
        u = k['uniform']

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

        tau_gases[15,0] = h2o[15] + o3[3] + u[5] + h2o_sq[15]
        tau_gases[16,0] = h2o[16] + o3[4] + u[6] + h2o_sq[16]

        tau_gases[17,0] = h2o[17] + o3[5] + u[7] + h2o_sq[17]
        tau_gases[18,0] = h2o[18] + o3[6] + u[8] + h2o_sq[18]

        tau_gases[19,0] = h2o[19] + o3[7] + u[9] + h2o_sq[19]
        tau_gases[20,0] = h2o[20] + o3[8] + u[10] + h2o_sq[20]

        tau_gases[21,0] = h2o_sq[21]
        tau_gases[22,0] = h2o_sq[22]

        tau_gases[23,0] = o3[9] + h2o_sq[23]
        tau_gases[24,0] = o3[10] + h2o_sq[24]

        tau_gases[25,0] = o3[11] + u[11] + h2o_sq[25]
        tau_gases[26,0] = o3[12] + u[12] + h2o_sq[26]

        tau_total = tau_gases + tau_lw + tau_iw
        input_1 = tau_lw / tau_total
        input_2 = tau_iw / tau_total

        input_3_direct = tau_total / mu
        input_3_diffuse = tau_total / mu_bar

        direct_ext = [net(input_1[k],input_2[k],input_3_direct[k], mu) for k, net in self.ext_net.items()]
        diffuse_ext = [net(input_1[k],input_2[k],input_3_diffuse[k], mu_bar) for k, net in self.ext_net.items()]

        direct_ext = tf.nn.softmax(direct_ext)
        diffuse_ext = tf.nn.softmax(diffuse_ext)

        direct_trans = tf.exp(-input_3_direct)
        diffuse_trans = tf.exp(-input_3_diffuse)

        return direct_ext, diffuse_ext, direct_trans, diffuse_trans
    
class WaterIceTransmissivityReflection_V1(Model):
    """
    This version computes cloud properties by explicitly modeling
    gamma and r_infinity
    """
    def __init__(self, n_hidden):
        super().__init__()

        # With real data these would be replaced with networks
        # that are functions of the water drop radius and possible
        # other factors
        self.gamma_water = [NoInputLayer(minval=0.0,maxval=2.0) for _ in np.arange(n_nets)]
        self.r_infinity_sq_water = [NoInputLayer(minval=0.0,maxval=1.0) for _ in np.arange(n_nets)]
        # RELU insures that tau is non-negative
        self.tau_water = [Dense(units=1, activation='relu', use_bias=False) for _ in np.arange(n_nets)]

        self.gamma_ice = [NoInputLayer(minval=0.0,maxval=2.0) for _ in np.arange(n_nets)]
        self.r_infinity_sq_ice = [NoInputLayer(minval=0.0,maxval=1.0) for _ in np.arange(n_nets)]
        # RELU insures that tau is non-negative
        self.tau_ice = [Dense(units=1, activation='relu', use_bias=False) for _ in np.arange(n_nets)]


    def call(self, lwp, iwp, mu):

        # Computing t, r, a for a cloud using 
        # eqs 13.65 and 13.66 in
        # "A first course in Atmospheric Radation"
        # 2nd Edition by Grant Petty, pages 409
        gamma_water = [net() for net in self.gamma_water]
        r_infinity_sq_water = [net() for net in self.r_infinity_sq_water]
        tau_water = [net(lwp) for net in self.tau_water]

        gamma_ice = [net() for net in self.gamma_ice]
        r_infinity_sq_ice = [net() for net in self.r_infinity_sq_ice]
        tau_ice = [net(iwp) for net in self.tau_ice]

        gamma_tau = gamma_water * tau_water
        e_plus = tf.exp(gamma_tau)
        e_minus = tf.exp(-gamma_tau)
        d = e_plus - r_infinity_sq_water * e_minus
        t_water = (1 - r_infinity_sq_water) / d
        r_water = r_infinity_sq_water * (e_plus - e_minus) / d

        gamma_tau = gamma_ice * tau_ice
        e_plus = tf.exp(gamma_tau)
        e_minus = tf.exp(-gamma_tau)
        d = e_plus - r_infinity_sq_ice * e_minus
        t_ice = (1 - r_infinity_sq_ice) / d
        r_ice = r_infinity_sq_ice * (e_plus - e_minus) / d

        # Treating ice and water as two sequential sub-layers
        # with multireflection. Using Adding / Doubling
        # method. See "A first course in Atmospheric Radation"
        # 2nd Edition by Grant Petty, pages 418-424

        t_direct = tf.exp(-(tau_water + tau_ice))
        # direct input, direct transmission
        t_direct_direct = t_direct / mu
        mu_diffuse = self.mu_diffuse_net()
        # diffuse input, direct transmision
        t_diffuse_direct = t_direct / mu_diffuse

        d = 1.0 - r_water * r_ice
        t_diffuse = t_water * t_ice / (d * mu_diffuse)


        # Taking average of the two possible sequences of layers
        # (water, ice) and (ice, water)
        r1 = r_water + t_water * t_water * r_ice / d
        r2 = r_ice + t_ice * t_ice * r_water / d
        r = 0.5 * (r1 + r2)

        a = 1.0 - t - r
        return t,r,a
    



class WaterIceTransmissivityReflection_V2(Model):
    """
    Version 2
    Computes cloud properties without explicity
    modeling any intermediate physical variables
    """
    def __init__(self, n_hidden):
        super().__init__()
        self.direct_net = [DenseFFN(n_hidden,3) for _ in n_nets]
        self.diffuse_net = [DenseFFN(n_hidden,2) for _ in n_nets]

    def _direct(self, lwp, iwp, mu):
        """
        Computes layer coefficients for direct input
        """
        X = tf.concat([lwp,iwp, mu],axis=-1)
        Y = [net(X) for net in self.direct_net]
        t_direct, t_diffuse, r = rearrange(Y)
        a = 1.0 - t_direct - t_diffuse - r
        return t_direct, t_diffuse, r, a

    def _diffuse(self, lwp, iwp):
        """
        Computes layer coefficients for diffuse input
        """
        X = tf.concat([lwp,iwp],axis=-1)
        Y = [net(X) for net in self.diffux_net]
        t, r = rearrange(Y)
        a = 1.0 - t - r
        return t, r, a

    def call(self, lwp, iwp, mu):
        direct_parameters = self._direct(lwp, iwp)
        diffuse_parameters = self._diffuse(lwp, iwp, mu)
        return direct_parameters, diffuse_parameters
    
class GasTransmissivity(Model):
    def __init__(self, n_hidden):
        super().__init__()
        self.tau_net = Tau(n_hidden)
        self.mu_diffuse_net = NoInputLayer(1)
    def call(self, temp, pressure, composition, lwp, iwp, mu):
        tau = self.tau_net(temp, pressure, composition, lwp, iwp)
        mu_diffuse = self.mu_diffuse_net()
        t_direct = [tf.math.exp(-v/mu) for v in tau]
        t_diffuse = [tf.math.exp(-v/mu_diffuse) for v in tau]
        r_fraction =
        return t_direct, t_diffuse
