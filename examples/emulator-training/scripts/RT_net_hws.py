import os

import numpy as np


import tensorflow as tf
from tensorflow.keras import losses, optimizers, layers, Input, Model, Layer, Sequential
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Dense,TimeDistributed

n_nets=53
class NoInputLayer(Model):
    def __init__(self,num_outputs,minval=0.0,maxval=1.0):
        super().__init__()
        self.w = tf.random.normal((num_outputs),minval=minval,maxval=maxval)
        self.w = tf.Variable(self.w)
    def call(self):
        return(self.w)

class DenseFFN(Model):
    """

    n_hidden[n_layers]: number of nodes
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

class WaterCoefficients(Model):
    """
    Network for representing gamma and r-infinity-squared for liquid 
    water and ice water.

    Normally the input would involve the effective radius of the 
    water drops / ice pellets
    
    n_hidden[n_layers]: number of nodes
    """
    def __init__(self, n_hidden):
        super().__init__()
        self.input = NoInputLayer(num_outputs=n_hidden[0])
        self.hidden = [Dense(units=n, activation='elu',) for n in n_hidden[1:]]
        # RELU insures that absorption coefficient is non-negative
        self.out = Dense(units=1, activation='relu') 

    def call(self):
        X = self.input()
        for hidden in self.hidden[1:]:
            X = hidden(X)
        return self.out(X)
    
class TauGas(Model):
    def __init__(self, n_hidden):
        super().__init__()

        n_k = {}
        n_k['h2o']=23
        n_k['o3'] =15
        n_k['co2']=15
        n_k['n2o']=5
        n_k['ch4']=15
        # 'u' represents all gases that are uniform in concentration in
        # the atmosphere: o2, n2
        # Could probably include co2, too, but it is separated out for
        # now to account for annual differences
        n_k['u']=15  

        self.ke = {}

        # Represents a function of temperature and pressure
        # used to build a gas absorption coefficient, ke 

        for gas,n in n_k.items():
            self.ke[gas] = [DenseFFN(n_hidden,1) for _ in np.arange(n)]

        # RELU insures that tau is non-negative
        self.tau = [Dense(units=1, activation='relu', use_bias=False) for _ in np.arange(n_nets)]

    # Note Ukkonen does not include nitrogen dioxide (no2) in simulation that generated data
    def call(self, temp, pressure, composition, lwp, iwp):
        X = tf.concat([temp,pressure],axis=-1)

        k = {}
        for gas, net in self.ke.items():
            k[gas] = [n(X) for n in net]
            k[gas] = tf.mult(k[gas],composition[gas])

        h2o = k['h2o']
        o3 = k['o3']
        co2 = k['co2']
        n2o = k['n2o']
        ch4 = k['ch4']
        u = k['uniform']

        x = []
        #0
        x.append(tf.concat([h2o[0],o3[0],co2[0],n2o[0],ch4[0],u[0],lwp[0],iwp[0]],axis=1))
        x.append(tf.concat([h2o[1],o3[1],co2[1],n2o[1],ch4[1],u[1],lwp[1],iwp[1]],axis=1))
        x.append(tf.concat([h2o[2],o3[2],co2[2],n2o[2],ch4[2],u[2],lwp[2],iwp[2]],axis=1))
        x.append(tf.concat([h2o[3],o3[3],co2[3],n2o[3],ch4[3],u[3],lwp[3],iwp[3]],axis=1))
        x.append(tf.concat([h2o[4],o3[4],co2[4],n2o[4],ch4[4],u[4],lwp[4],iwp[4]],axis=1))
        #5
        x.append(tf.concat([h2o[5],o3[5],u[5]],axis=1))
        x.append(tf.concat([h2o[6],o3[6],u[6]],axis=1))
        x.append(tf.concat([h2o[7],o3[7]],axis=1))
        x.append(tf.concat([h2o[8],o3[8]],axis=1))
        x.append(tf.concat([h2o[9],u[7]],axis=1))
        #10
        x.append(tf.concat([h2o[10],u[8]],axis=1))
        x.append(tf.concat([o3[9],u[9]],axis=1))
        x.append(tf.concat([o3[10],u[10]],axis=1))
        x.append(tf.concat([h2o[11]],axis=1))
        x.append(tf.concat([h2o[12]],axis=1))
        #15
        x.append(tf.concat([h2o[13],ch4[5]],axis=1))
        x.append(tf.concat([h2o[14],ch4[6]],axis=1))
        x.append(tf.concat([h2o[15],ch4[7]],axis=1))
        x.append(tf.concat([h2o[16],co2[5]],axis=1))
        x.append(tf.concat([h2o[17],co2[6]],axis=1))
        #20
        x.append(tf.concat([h2o[18],co2[7]],axis=1))
        x.append(tf.concat([ch4[8]],axis=1))
        x.append(tf.concat([ch4[9]],axis=1))
        x.append(tf.concat([ch4[10]],axis=1))
        x.append(tf.concat([co2[8]],axis=1))
        #25
        x.append(tf.concat([co2[9]],axis=1))
        x.append(tf.concat([co2[10]],axis=1))
        x.append(tf.concat([o3[11]],axis=1))
        x.append(tf.concat([o3[12]],axis=1))
        x.append(tf.concat([u[11]],axis=1))
        #30
        x.append(tf.concat([lwp[5]],axis=1))
        x.append(tf.concat([lwp[6]],axis=1))
        x.append(tf.concat([iwp[5]],axis=1))
        x.append(tf.concat([iwp[6]],axis=1))
        x.append(tf.concat([lwp[7],iwp[7]],axis=1))
        #35
        x.append(tf.concat([lwp[8],iwp[8]],axis=1))
        x.append(tf.concat([lwp[9],h2o[19]],axis=1))
        x.append(tf.concat([lwp[10],h2o[20]],axis=1))
        x.append(tf.concat([iwp[9],h2o[21]],axis=1))
        x.append(tf.concat([iwp[10],h2o[22]],axis=1))
        #40
        x.append(tf.concat([lwp[11],co2[11]],axis=1))
        x.append(tf.concat([lwp[12],co2[12]],axis=1))
        x.append(tf.concat([iwp[11],co2[13]],axis=1))
        x.append(tf.concat([iwp[12],co2[14]],axis=1))
        x.append(tf.concat([lwp[13],ch4[11]],axis=1))
        #45
        x.append(tf.concat([lwp[14],ch4[12]],axis=1))
        x.append(tf.concat([iwp[13],ch4[13]],axis=1))
        x.append(tf.concat([iwp[14],ch4[14]],axis=1))
        x.append(tf.concat([lwp[15],o3[13]],axis=1))
        x.append(tf.concat([lwp[16],o3[14]],axis=1))
        #50
        x.append(tf.concat([iwp[15],u[12]],axis=1))
        x.append(tf.concat([iwp[16],u[13]],axis=1))
        x.append(tf.concat([u[14]],axis=1))

        if len(x) != len(self.tau): 
            raise Exception(f"len of self.tau = {len(self.tau)} is inconsistent with number of inputs = {len(x)}")

        tau = [net(x[i]) for i,net in self.tau.items()]
        return tau
    
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
