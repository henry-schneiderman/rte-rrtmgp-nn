import os

import numpy as np


import tensorflow as tf
from tensorflow.keras import losses, optimizers, layers, Input, Model, Layer, Sequential
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Dense,TimeDistributed

class NoInputLayer(Model):
    def __init__(self,num_outputs):
        super().__init__()
        self.w = tf.random.normal((num_outputs),minval=0.0,maxval=1.0)
        self.w = tf.Variable(self.w)
    def call(self):
        return(self.w)

class Ke(Model):
    """
    Represents a function of temperature and pressure
    used to build absorption coefficient, ke 
    
    n_hidden[n_layers]: number of nodes
    """
    def __init__(self, n_hidden):
        super().__init__()
        self.hidden = [Dense(units=n, activation='elu',) for n in n_hidden]
        # RELU insures that absorption coefficient is non-negative
        self.out = Dense(units=1, activation='relu') 

    def call(self, X):
        for hidden in self.hidden:
            X = hidden(X)
        return self.out(X)
    
class Tau(Model):
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

        self.k = {}

        for gas,n in n_k.items():
            self.k[gas] = [Ke(n_hidden) for _ in np.arange(n)]

        n_nets=53
        # RELU insures that tau is non-negative
        self.tau = [Dense(units=1, activation='relu', use_bias=False) for _ in np.arange(n_nets)]

    # Note Ukkonen does not include nitrogen dioxide (no2) in simulation that generated data
    def call(self, temp, pressure, composition, lwp, iwp):
        X = tf.concat([temp,pressure],axis=-1)

        k = {}
        for gas, net in self.k.items():
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


class Transmissivity(Model):
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
