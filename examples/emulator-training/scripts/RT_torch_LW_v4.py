# 

# Same as RT_torch_LW.v1.py except penalizes for clear sky
# Same as RT_torch_LW.v2.py except computes sources differently
# using radiation_two_streams.F90 lines 310+
# Same as RT_torch_LW.v3.py except using adding-doubling algorithm
# from radiation_adding_ica_lw.F90!

import numpy as np
import time
from typing import List
import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F

from RT_data_hws import absorbed_flux_to_heating_rate
import RT_lw_data

# all versions up to and including v19, v24
eps_1 = 0.0000001
#v22
#eps_1 = 0.002
#v23
#eps_1 = 0.01

t_direct_scattering = 0.0
t_direct_split = 0.0
t_scattering_v2_tau = 0.0
t_extinction = 0.0
t_total = 0.0
t_train = 0.0
t_loss = 0.0
t_grad = 0.0
t_backward = 0.0

class MLP(nn.Module):
    """
    Multi Layer Perceptron (MLP) module

    Fully connected layers
    
    Uses ReLU() activation for hidden units
    No activation for output unit
    
    Initialization of all weights with uniform distribution with 'lower' 
    and 'upper' bounds. Defaults to -0.1 < weight < 0.1
    
    Hidden units initial bias with uniform distribution 0.9 < x < 1.1
    Output unit initial bias with uniform distribution -0.1 < x <0.1
    """

    def __init__(self, n_input, n_hidden: List[int], n_output, dropout_p, device, 
                 lower=-0.1, upper=0.1, bias=True):
        super(MLP, self).__init__()
        self.n_hidden = n_hidden
        self.n_outputs = n_output
        n_last = n_input
        self.hidden = nn.ModuleList()

        for n in n_hidden:
            mod = nn.Linear(n_last, n, bias=bias,device=device)
            torch.nn.init.uniform_(mod.weight, a=lower, b=upper)
            # Bias initialized to ~1.0
            # Because of ReLU activation, don't want any connections to
            # be prematurely pruned away by becoming negative.
            # Therefore start with a significant positive bias
            if bias:
                torch.nn.init.uniform_(mod.bias, a=0.9, b=1.1) #a=-0.1, b=0.1)
            self.hidden.append(mod)
            n_last = n
        self.dropout_p = dropout_p
        self.output = nn.Linear(n_last, n_output, bias=bias, device=device)
        torch.nn.init.uniform_(self.output.weight, a=lower, b=upper)
        if bias:
            torch.nn.init.uniform_(self.output.bias, a=-0.1, b=0.1)

    def reset_dropout(self,dropout_p):
        self.dropout_p = dropout_p

    #@torch.compile
    def forward(self, x):

        for hidden in self.hidden:
            x = hidden(x)
            x = F.relu(x)
            x = F.dropout(x,p=self.dropout_p,training=self.training)
        return self.output(x)

class BD(nn.Module):
    """
    Block Diagonal (BD) module

    """

    def __init__(self, n_input, n_hidden, n_output, dropout_p, device, bias=False): 

        super(BD, self).__init__()
        self.n_hidden = n_hidden
        self.dropout_p = dropout_p
       
        weight_values = torch.rand((n_input, self.n_hidden[0]),
                                   requires_grad=True,device=device,
                                   dtype=torch.float32,)
        
        self.input_weight = nn.parameter.Parameter(weight_values, requires_grad=True)

        self.bias = bias
        if bias:
            bias_values = torch.rand((self.n_hidden[0],),
                                   requires_grad=True,device=device,
                                   dtype=torch.float32,)
            self.input_bias = nn.parameter.Parameter(bias_values, requires_grad=True)

            biases = []

        # v12
        #template = torch.ones((6,6), device=device,
        #                           dtype=torch.float32,)

        #v13
        template = torch.ones((8,8), device=device,
                                   dtype=torch.float32,)
        self.filter = torch.block_diag (template,template,template,template)

        weights = []

        n_last = n_hidden[0]
        for n in n_hidden[1:]:
            weights.append(torch.rand((n_last, n),
                                   requires_grad=True,device=device,
                                   dtype=torch.float32,))

            if bias:
                biases.append(torch.rand((n,),
                                   requires_grad=True,device=device,
                                   dtype=torch.float32,))
        
            #weights.append(nn.parameter.Parameter(weight_values, requires_grad=True))
            n_last = n
        self.weights = torch.nn.ParameterList(weights)


        tmp_weights = torch.rand((n_last, n_output),
                                   requires_grad=True,device=device,
                                   dtype=torch.float32,)

        self.output_weights = nn.parameter.Parameter(tmp_weights, requires_grad=True)

        if bias:
            self.biases = torch.nn.ParameterList(biases)
            tmp_weights = torch.rand((n_output,),
                                    requires_grad=True,device=device,
                                    dtype=torch.float32,)

            self.output_bias = nn.parameter.Parameter(tmp_weights, requires_grad=True)

        template2 = torch.ones((4,3), device=device,
                                   dtype=torch.float32,)
        self.output_filter = torch.block_diag (template2,template2,template2,template2, template2,template2,template2,template2)
            
    def reset_dropout(self,dropout_p):
        self.dropout_p = dropout_p

    #@torch.compile
    def forward(self, x):

        if self.bias:
            x = x @ self.input_weight + self.input_bias
            x = F.relu(x)
            x = F.dropout(x,p=self.dropout_p,training=self.training)
            for i, weight in enumerate(self.weights):
                x = x @ (self.filter * weight) + self.biases[i]
                x = F.relu(x)
                x = F.dropout(x,p=self.dropout_p,training=self.training)
            x = x @ (self.output_filter * self.output_weights) + self.output_bias
        else:
            x = x @ self.input_weight
            x = F.relu(x)
            x = F.dropout(x,p=self.dropout_p,training=self.training)
            for weight in self.weights:
                x = x @ (self.filter * weight)
                x = F.relu(x)
                x = F.dropout(x,p=self.dropout_p,training=self.training)
            x = x @ (self.output_filter * self.output_weights)
        return x

class LayerDistributed(nn.Module):
    """
    Applies a nn.Module independently to an array of atmospheric layers

    Same idea as TensorFlow's TimeDistributed Class
    
    Adapted from:
    https://stackoverflow.com/questions/62912239/tensorflows-timedistributed-equivalent-in-pytorch

    The input and output may each be a single
    tensor or a list of tensors.

    Each tensor has dimensions: (n_samples, n_layers, data's dimensions. . .)
    """

    def __init__(self, module):
        super(LayerDistributed, self).__init__()
        self.module = module

    def reset_dropout(self,dropout_p):
        self.module.reset_dropout(dropout_p)

    def forward(self, x):
        if torch.is_tensor(x):
            shape = x.shape
            n_sample = shape[0]
            n_layer = shape[1]
            squashed_input = x.contiguous().view(n_sample*n_layer, *shape[2:]) 
        else: 
            # else 'x' is a list of tensors. Squash each individually
            squashed_input = []
            for xx in x:
                # Squash samples and layers into a single dimension
                shape = xx.shape
                n_sample = shape[0]
                n_layer = shape[1]
                xx_reshape = xx.contiguous().view(n_sample*n_layer, *shape[2:])
                squashed_input.append(xx_reshape)
        y = self.module(squashed_input)
        # Reshape y
        if torch.is_tensor(y):
            shape = y.shape
            unsquashed_output = y.contiguous().view(n_sample, n_layer, 
                                                    *shape[1:]) 
        else:
            # else 'y' is a list of tensors. Unsquash each individually
            unsquashed_output = []
            for yy in y:
                shape = yy.shape
                yy_reshaped = yy.contiguous().view(n_sample, n_layer, 
                                                   *shape[1:])
                unsquashed_output.append(yy_reshaped)
        return unsquashed_output


class Extinction(nn.Module):
    """ 
    Generates optical depth for each atmospheric 
    constituent for each channel for the given layer.
    
    Learns the mass extinction coefficients and the dependence 
    of these coefficients on temperature and pressure.

    Hard-codes the multiplication of each mass
    extinction coefficient by the consistuent's mass

    Inputs:
        Mass of each atmospheric constituent
        Temperature, pressure, and log_pressure

    Outputs
        Optical depth of each constituent in each channel
    """
    def __init__(self, n_channel, dropout_p, device):
        super(Extinction, self).__init__()
        self.n_channel = n_channel
        self.device = device
        self.dropout_p = dropout_p

        # Computes a scalar extinction coeffient for each constituent 
        # for each channel
        self.net_lw  = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_iw  = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_h2o = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_o3  = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_co2 = nn.Linear(1,self.n_channel,bias=False,device=device)
        
        self.net_o2   = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_n2o = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_ch4 = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_co = nn.Linear(1,self.n_channel,bias=False,device=device)

        n_weights = 8 * n_channel

        lower = -0.9 # exp(-0.9) = .406
        upper = 0.5  # exp(0.5) = 1.64
        torch.nn.init.uniform_(self.net_lw.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_iw.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_h2o.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_o3.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_co2.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_o2.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_n2o.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_ch4.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_co.weight, a=lower, b=upper)

        # exp() activation forces extinction coefficient to always be positive
        # and never negative or zero

        # Modifies each extinction coeffient as a function of temperature, 
        # pressure 
        # Seeks to model pressuring broadening of atmospheric absorption lines
        # Single network for each constituent
        self.net_ke_h2o = MLP(n_input=2,n_hidden=(6,4,4),n_output=1,
                            dropout_p=dropout_p,device=device)
        self.net_ke_o3  = MLP(n_input=2,n_hidden=(6,4,4),n_output=1,
                              dropout_p=dropout_p,device=device)
        self.net_ke_co2 = MLP(n_input=2,n_hidden=(6,4,4),n_output=1,
                              dropout_p=dropout_p,device=device)
        self.net_ke_o2   = MLP(n_input=2,n_hidden=(6,4,4),n_output=1,
                              dropout_p=dropout_p,device=device)
        self.net_ke_n2o = MLP(n_input=2,n_hidden=(6,4,4),n_output=1,
                              dropout_p=dropout_p,device=device)
        self.net_ke_ch4 = MLP(n_input=2,n_hidden=(6,4,4),n_output=1,
                              dropout_p=dropout_p,device=device)
        self.net_ke_co = MLP(n_input=2,n_hidden=(6,4,4),n_output=1,
                              dropout_p=dropout_p,device=device)

        n_weights += 7 * (12 + 24 + 16 + 4 + 6 + 4 + 4 + 1)
        print(f"Extinction n weights = {n_weights}")


        # Filters select which channels each constituent contributes to
        # Follows similiar assignment of bands as
        # Table A2 in Pincus, R., Mlawer, E. J., &
        # Delamere, J. S. (2019). Balancing accuracy, efficiency, and 
        # flexibility in radiation calculations for dynamical models. Journal 
        # of Advances in Modeling Earth Systems, 11,3074–3089. 
        # https://doi.org/10.1029/2019MS001621

        #self.filter_h2o = torch.tensor([1,1,1,1,1, 1,1,1,1,1, 1,0,0,0,],
        #                               dtype=torch.float32,device=device)

        filter_h2o = torch.tensor([1,1,1,1,1, 1,1,1,1,1, 1,1,1,0,1, 1],
                                       dtype=torch.float32,device=device)
        
        self.filter_h2o = torch.cat([filter_h2o, filter_h2o, filter_h2o])

        filter_o3 = torch.tensor([0,0,0,1,1, 0,1,1,0,0, 0,0,1,0,0, 0],
                                       dtype=torch.float32,device=device)

        self.filter_o3 = torch.cat([filter_o3, filter_o3, filter_o3])

        filter_co2 = torch.tensor([0,0,1,1,1, 1,1,1,0,0, 0,1,1,1,1, 0],
                                       dtype=torch.float32,device=device)
        
        self.filter_co2 = torch.cat([filter_co2, filter_co2, filter_co2])
        
        # used for both o2 and n2 since they are both constants
        filter_o2  = torch.tensor([1,0,0,0,0, 0,0,0,0,0, 1,0,0,0,0, 0],
                                       dtype=torch.float32,device=device)
        
        self.filter_o2 = torch.cat([filter_o2, filter_o2, filter_o2])

        filter_n2o = torch.tensor([0,0,1,0,0, 0,0,1,1,0, 0,0,1,0,1, 0],
                                       dtype=torch.float32,device=device)

        self.filter_n2o = torch.cat([filter_n2o, filter_n2o, filter_n2o])
        
        filter_ch4 = torch.tensor([0,0,0,0,0, 0,0,0,1,0, 0,0,0,0,0, 1,],
                                       dtype=torch.float32,device=device)
        
        self.filter_ch4 = torch.cat([filter_ch4, filter_ch4, filter_ch4])

        filter_co = torch.tensor([0,0,0,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,],
                                       dtype=torch.float32,device=device)
        
        self.filter_co = torch.cat([filter_co, filter_co, filter_co])


        n_weights = (n_weights - 7 * n_channel + torch.sum(self.filter_ch4)
                     + torch.sum(self.filter_o3) + torch.sum(self.filter_co2)
                     + torch.sum(self.filter_o2) + torch.sum(self.filter_n2o)
                     + torch.sum(self.filter_h2o) + torch.sum(self.filter_co))

        print(f"Extinction trainable weights = {n_weights}")

    def reset_dropout(self,dropout_p):
        self.dropout_p = dropout_p
        self.net_ke_h2o.reset_dropout(dropout_p)
        self.net_ke_o3.reset_dropout(dropout_p)
        self.net_ke_co2.reset_dropout(dropout_p)
        self.net_ke_o2.reset_dropout(dropout_p)
        self.net_ke_n2o.reset_dropout(dropout_p)
        self.net_ke_ch4.reset_dropout(dropout_p)
        self.net_ke_co.reset_dropout(dropout_p)

    def forward(self, x):

        temperature_pressure, constituents = x

        c = constituents
        shape = c.shape
        c = c.reshape((shape[0],1,shape[1]))
        t_p = temperature_pressure

        #def d(value):
        #    return F.dropout(value,p=self.dropout_p,training=self.training)
        
        a = torch.exp
        b = torch.sigmoid

        one = torch.ones((shape[0],1),dtype=torch.float32,device=self.device)
        # a(self.net_lw (one)): (n_examples, n_channels)
        tau_lw  = a(self.net_lw (one)) * (c[:,:,0])
        tau_iw  = a(self.net_iw (one)) * (c[:,:,1])
        tau_h2o = a(self.net_h2o(one)) * (c[:,:,2]) * (self.filter_h2o * 
                                                       b(self.net_ke_h2o(t_p)))
        tau_o3  = a(self.net_o3 (one)) * (c[:,:,3] * (self.filter_o3  * 
                                                       b(self.net_ke_o3 (t_p))))
        tau_co2 = a(self.net_co2(one)) * (c[:,:,4]) * (self.filter_co2 * 
                                                       b(self.net_ke_co2(t_p)))
        tau_o2   = a(self.net_o2  (one)) * (c[:,:,5]) * (self.filter_o2 * 
                                                       b(self.net_ke_o2  (t_p)))
        tau_n2o = a(self.net_n2o(one)) * (c[:,:,6]) * (self.filter_n2o * 
                                                       b(self.net_ke_n2o(t_p)))
        tau_ch4 = a(self.net_ch4(one)) * (c[:,:,7]) * (self.filter_ch4 * 
                                                       b(self.net_ke_ch4(t_p)))

        tau_co = a(self.net_co(one)) * (c[:,:,8]) * (self.filter_co * 
                                                       b(self.net_ke_co(t_p)))


        tau_lw  = torch.unsqueeze(tau_lw,2)
        tau_iw  = torch.unsqueeze(tau_iw,2)
        tau_h2o = torch.unsqueeze(tau_h2o,2)
        tau_o3  = torch.unsqueeze(tau_o3,2)
        tau_co2 = torch.unsqueeze(tau_co2,2)

        tau_o2   = torch.unsqueeze(tau_o2,2)
        tau_n2o = torch.unsqueeze(tau_n2o,2)
        tau_ch4 = torch.unsqueeze(tau_ch4,2)
        tau_co = torch.unsqueeze(tau_co,2)

        tau = torch.cat([tau_lw, tau_iw, tau_h2o, tau_o3, tau_co2, tau_o2, 
                         tau_n2o, tau_ch4, tau_co],dim=2)

        return tau

    
def tensorize(np_ndarray):
    t = torch.from_numpy(np_ndarray).float()
    return t

class Scattering_v2_tau_efficient(nn.Module):
    """ 
    Same as V1_tau except m scattering nets.
    Each channel outputs its own weighted combo of the scattering nets
    Similar to Scattering_v2_tau, except implemented for efficiency

    """

    def __init__(self, n_channel, n_constituent, dropout_p, device):

        super(Scattering_v2_tau_efficient, self).__init__()
        self.n_channel = n_channel
        #self.n_scattering_nets = 5 # settings for 8
        self.n_scattering_nets = 8 # settings for 11

        n_input = n_constituent #5

        tmp_array = np.ones((n_constituent), dtype=np.float32)
        tmp_array[0] = 0.0
        tmp_array[1] = 0.0
        tmp_array = tmp_array.reshape((1,1,-1))
        self.clear_sky_mask = tensorize(tmp_array).to(device)

        n_hidden = [32, 32, 32] 

        # Create basis functions for scattering

        self.diffuse_scattering = BD(n_input=n_input, 
                                    n_hidden=n_hidden, 
                                    n_output=24,
                                    dropout_p=dropout_p,
                                    device=device,
                                    bias=True) 

        # Select combo of basis to give a,r,t


        self.diffuse_selection = nn.Conv2d(in_channels=self.n_channel,
                                          out_channels=self.n_channel, 
                                          kernel_size=(self.n_scattering_nets,1), 
                                          stride=(1,1), padding=0, dilation=1, 
                                          groups=self.n_channel, bias=False, device=device)

        n_weights = n_input * n_hidden[0] + n_hidden[0]*n_hidden[1]
        n_weights += n_hidden[1]*n_hidden[2] + n_hidden[2]*3*self.n_scattering_nets
        n_weights += n_hidden[0] + n_hidden[1] + n_hidden[2] + 3*self.n_scattering_nets
        n_weights = n_weights * 2 + n_hidden[0]
        n_weights += self.n_scattering_nets * self.n_channel * 2
        print(f"Scattering_tau_2_efficient number of weights = {n_weights}")

        n_weights = n_input * n_hidden[0] + 64 * 4 + 64 * 4 
        n_weights += 12 * 8 
        n_weights += n_hidden[0] + n_hidden[1] + n_hidden[2] + 3*self.n_scattering_nets
        n_weights = n_weights * 2 + n_hidden[0]
        n_weights += self.n_scattering_nets * self.n_channel * 2
        print(f"Scattering_tau_2_efficient number of learned weights = {n_weights}")

    def reset_dropout(self,dropout_p):


        self.diffuse_scattering.reset_dropout(dropout_p)


    def forward(self, x):
        (tau, ) = x

        #print(f"tau.shape = {tau.shape}")
        # sum over constituents
        tau_full_total = torch.sum(tau, dim=2, keepdims=False)

        tau_clear_total = torch.sum(tau[:,:,2:], dim=2, keepdims=False)


        t_full = torch.exp(-tau_full_total)
        t_clear = torch.exp(-tau_clear_total)

        # f = number of features
        # [i,channels,f]

        e_split_full = self.diffuse_scattering(tau)
        n = e_split_full.shape[0]
        
        # [i,channels,3, m]


        e_split_full = torch.reshape(e_split_full,
                                       (n, self.n_channel,
                                        self.n_scattering_nets, 3))
        # [i,channels, m, 3]
        e_split_full = F.softmax(e_split_full,dim=-1)

        e_split_full = self.diffuse_selection(e_split_full)

        # [i, channels, 1, 3]

        e_split_full = torch.squeeze(e_split_full, dim=-2)

        # [i,channels,3]  

        e_split_full = F.softmax(e_split_full, dim=-1)

        # Repeat for clear case

        tau_clear = self.clear_sky_mask * tau
        # f = number of features
        # [i,channels,f]

        e_split_clear = self.diffuse_scattering(tau_clear)
        n = e_split_clear.shape[0]
        
        # [i,channels,3, m]


        e_split_clear = torch.reshape(e_split_clear,
                                       (n, self.n_channel,
                                        self.n_scattering_nets, 3))
        # [i,channels, m, 3]
        e_split_clear = F.softmax(e_split_clear,dim=-1)

        e_split_clear = self.diffuse_selection(e_split_clear)

        # [i, channels, 1, 3]

        e_split_clear = torch.squeeze(e_split_clear, dim=-2)

        # [i,channels,3]  

        e_split_clear = F.softmax(e_split_clear, dim=-1)

        layers = [t_full, e_split_full, tau_full_total, t_clear, e_split_clear, tau_clear_total]

        return layers



class MultiReflection(nn.Module):
    """ 
    Recomputes each layer's radiative coefficients by accounting
    for interaction (multireflection) with all other layers using the 
    Adding-Doubling method (no learning).
    """

    def __init__(self, n_channels, n_bands, device):
        super(MultiReflection, self).__init__()
   
        self.device = device

        weight_values = torch.rand((n_bands, n_channels),
                                   requires_grad=True,device=device,
                                   dtype=torch.float32,)
        
        diagonal = torch.full((n_bands,), 1.2, dtype=torch.float32,device=device)
        
        template = torch.diag(diagonal)

        offset = torch.cat((template, template, template),dim=1)
        
        self.bands_to_channels = nn.parameter.Parameter(weight_values * 0.5 + offset, requires_grad=True)

    def _compute_upward(self,r,t, os_up, os_down):
        """

        Bottom up computation of cumulative surface reflection
        and denominator factor

        n = len(r)
        Output has same size

        rs index corresponds to newly formed virtual surface; coincides with r
        ds index corresponds to " "

        """

        # n = len(r)
        # Start at bottom of the atmosphere (layer = n-1)
        rs = []
        last_rs = r[:,-1,:]  # reflectance of original surface is unchanged

        rs.append(last_rs) #
        ds = []
        shape = r.shape
        ds.append(torch.ones((shape[0],shape[2]), dtype=torch.float32,device=self.device)) #

        s = []
        last_s = os_up[:,-1,:] #
        s.append(last_s)

        # n-2 . . 0 (inclusive)
        for l in reversed(torch.arange(start=0, end=r.shape[1]-1, device=self.device)):
            dd = 1.0 / (1.0 - last_rs * r[:,l,:])  #
            last_s = os_up[:,l,:] + t[:,l,:] * (last_s + last_rs * os_down[:,l,:]) * dd  #
            s.append(last_s)

            ds.append(dd)
            last_rs = r[:,l,:] + t[:,l,:] * t[:,l,:] * last_rs * dd #
            rs.append(last_rs)


        rs = torch.stack(rs,dim=1)  
        ds = torch.stack(ds,dim=1)  
        s = torch.stack(s,dim=1)  

        rs = torch.flip(rs, dims=(1,))  # n values: 0 .. n-1 (includes surface)
        ds = torch.flip(ds, dims=(1,)) # n values: 0 .. n-1 (last value is one)
        s = torch.flip(s, dims=(1,))

        return rs, ds, s
    
   
            
    def _compute_downward (self, ss, s_down, rs, ds, r, t):
        """ 
        Input
            s_down represents layer it is exiting
            n-1 = len(s_down)

        Output
            index of flux represents radiation flowing into layer
            n = len(output)
        """

        # no downward flux or absorption for layer=0
        shape = t.shape
        flux = torch.zeros((shape[0],shape[2]), dtype=torch.float32, device=self.device)
        flux_down = []
        flux_down.append(flux)

        flux_up = []
        flux_up.append(ss[:,0,:])

        for l in torch.arange(0, t.shape[1]-1, device=self.device):
            flux = t[:,l,:] * flux  + r[:,l,:] * ss[:,l+1,:] + s_down[:,l,:] * ds[:,l,:]

            flux_down.append(flux)

            flux_up.append(flux * rs[:,l+1,:] + ss[:,l+1,:])


        flux_down = torch.stack(flux_down, dim=1)
        flux_up = torch.stack(flux_up, dim=1)


        # n output values
        return flux_down, flux_up


    
    def forward(self, x):
        """
        Computations are independent across channel.

        The prefixes -- t, e, r, a -- correspond respectively to
        transmission, extinction, reflection, and absorption.
        """

        x_sources, layers, x_emissivity = x

        t_full, e_split_full, tau = layers

        shape = t_full.shape


        e = 1.0 - t_full

        t = t_full + e * e_split_full[:,:,:,0]
        r = e * e_split_full[:,:,:,1]
        a = e * e_split_full[:,:,:,2]

        #diff = a + r + t - torch.ones((1,1,1), dtype=torch.float32, device=self.device)

        #if torch.max(torch.abs(diff)) > 0.01:
        #    print(f"Max deviation from 1.0 = {torch.max(torch.abs(diff))}")

        (r_surface, a_surface) = (1.0 - x_emissivity, x_emissivity)
        r_surface = r_surface.reshape((-1,1,1))
        a_surface = a_surface.reshape((-1,1,1))
        t_surface = torch.zeros((1,1,1), dtype=torch.float32, device=self.device)
        r_surface = r_surface.expand(shape[0],1,shape[2])
        a_surface = a_surface.expand(shape[0],1,shape[2])
        t_surface = t_surface.expand(shape[0],1,shape[2])

        # (n_examples, n_half_levels + 1, n_bands) * (n_bands, n_channels)
        x_sources = torch.matmul(x_sources, F.softmax(self.bands_to_channels, dim=1)) 

        surface_source = x_sources[:,-1:,:]
        # exclude surface
        hl_sources = x_sources[:,:-1,:]

        #[:,n_levels,:]
        # avoid division by zero
        coef = (hl_sources[:,1:,:] - hl_sources[:,:-1,:]) / (tau + 1.0e-06)

        coeff_up_top = coef + hl_sources[:,:-1,:]
        coeff_up_bot = coef + hl_sources[:,1:,:]

        coeff_dn_top  = -coef + hl_sources[:,:-1,:]
        coeff_dn_bot  = -coef + hl_sources[:,1:,:]

        s_up = coeff_up_top - r * coeff_dn_top - t * coeff_up_bot
        s_dn = coeff_dn_bot - r * coeff_up_bot - t * coeff_dn_top

        s_both = a * 0.5 * (hl_sources[:,1:,:] + hl_sources[:,:-1,:]) 

        mask = tau > 1.0e-03

        s_up = s_up * mask + s_both * ~mask
        s_dn = s_dn * mask + s_both * ~mask

        s_surface = surface_source * a_surface

        os_up = torch.cat((s_up,s_surface),axis=1)
        os_dn = torch.cat((s_dn,s_surface),axis=1)

        #a = torch.cat((a, a_surface), dim=1)
        r = torch.cat((r, r_surface), dim=1)
        t = torch.cat((t, t_surface), dim=1)
        #total_input = 2.0 * torch.sum(s[:,:-1,:]) + torch.sum(s[:,-1,:]) 

        rs, ds, ss = self._compute_upward(r,t, os_up, os_dn)

        # output: n values; flow into layer; index corresponds to entering layer
        flux_down, flux_up = self._compute_downward (ss, os_dn, rs, ds, r, t)

        #remaining_flux = torch.sum(flux_up[:,0,:])

        #total_absorbed_flux = torch.sum(absorbed_flux)

        #total_output = remaining_flux + total_absorbed_flux

        #diff = torch.abs(total_input - total_output)
        #if diff > 10.0:
        #    print (f"loss of energy = {diff}")

        return flux_down, flux_up

class FullNet(nn.Module):
    """ Computes full radiative transfer (direct and diffuse radiation)
    for an atmospheric column """

    def __init__(self, n_channel, n_constituent, n_band, dropout_p, device):
        super(FullNet, self).__init__()
        self.device = device
        self.n_channel = n_channel

        # Learns optical depth for each layer for each constituent for 
        # each channel
        self.extinction_net = LayerDistributed(Extinction(n_channel,dropout_p,
                                                          device))
        
        self.scattering_net = LayerDistributed(Scattering_v2_tau_efficient(n_channel,
                                                    n_constituent,
                                                    dropout_p,
                                                    device))

        self.multireflection_net = MultiReflection(n_channel,n_band,device)


    def reset_dropout(self,dropout_p):
        self.extinction_net.reset_dropout(dropout_p)
        self.scattering_net.reset_dropout(dropout_p)

    def forward(self, x):

        x_layers, x_sources, x_emissivity,  _, _ = x

        #print(f"x_layers.shape = {x_layers.shape}")
        #9 constituents: lwc, ciw, h2o, o3, co2,  o2, n2o, ch4, co,  -no2?, 
        (temperature_pressure, 
        constituents) = (x_layers[:,:,0:2], 
                        x_layers[:,:,2:11])
    
        tau = self.extinction_net((temperature_pressure, 
                                constituents))

        layers = self.scattering_net((tau,))

        t_full, e_split_full, tau_full, t_clear, e_split_clear, tau_clear = layers

        flux_full = self.multireflection_net([x_sources, [t_full, e_split_full, tau_full], x_emissivity])

        flux_clear = self.multireflection_net([x_sources, [t_clear, e_split_clear, tau_clear], x_emissivity])

        flux_down_full, flux_up_full = flux_full
        flux_down_clear, flux_up_clear = flux_clear

        flux_down_full = torch.sum(flux_down_full,dim=2)
        flux_up_full = torch.sum(flux_up_full,dim=2)

        flux_down_clear = torch.sum(flux_down_clear,dim=2)
        flux_up_clear = torch.sum(flux_up_clear,dim=2)

        flux = (flux_down_full, flux_up_full, flux_down_clear, flux_up_clear)

        return flux

def loss_energy(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred):
    
    flux_absorbed_true = (flux_down_true[:,:-1] -
                             flux_down_true[:,1:] + 
                             flux_up_true[:,1:] -
                             flux_up_true[:,:-1])

    flux_absorbed_true = (torch.sum(flux_absorbed_true, dim=(0,1)) + 
    torch.sum(flux_down_true[:,-1] - flux_up_true[:,-1] + flux_up_true[:,0]))

    flux_absorbed_pred = (flux_down_pred[:,:-1] -
                             flux_down_pred[:,1:] + 
                             flux_up_pred[:,1:] -
                             flux_up_pred[:,:-1])

    flux_absorbed_pred = (torch.sum(flux_absorbed_pred, dim=(0,1)) +
    torch.sum(flux_down_pred[:,-1] - flux_up_pred[:,-1] + flux_up_pred[:,0]))

    shape = flux_down_true.shape

    return (flux_absorbed_pred - flux_absorbed_true) / (shape[0] * shape[1])



def loss_heating_rate(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred,
                           delta_pressure):
    
    flux_absorbed_true = (flux_down_true[:,:-1] -
                             flux_down_true[:,1:] + 
                             flux_up_true[:,1:] -
                             flux_up_true[:,:-1])

    flux_absorbed_pred = (flux_down_pred[:,:-1] -
                             flux_down_pred[:,1:] + 
                             flux_up_pred[:,1:] -
                             flux_up_pred[:,:-1])
    heat_true = absorbed_flux_to_heating_rate(flux_absorbed_true, 
                                              delta_pressure)
    heat_pred = absorbed_flux_to_heating_rate(flux_absorbed_pred, 
                                              delta_pressure)
    loss = torch.sqrt(torch.mean(torch.square(heat_true - heat_pred),
                                  dim=(0,1),keepdim=False))
    return loss


def loss_clear_heating_rate_wrapper(data, y_pred):
    _, _, _, delta_pressure, y_true = data

    (_, _, flux_down_pred, flux_up_pred) = y_pred
    
    (flux_down_true, flux_up_true) = (y_true[:,:,2], y_true[:,:,3])
                         
    hr_loss = loss_heating_rate(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred, delta_pressure)
    
    return hr_loss

def loss_full_heating_rate_wrapper(data, y_pred):
    _, _, _, delta_pressure, y_true = data

    (flux_down_pred, flux_up_pred, _, _) = y_pred
    (flux_down_true, flux_up_true) = (y_true[:,:,0], y_true[:,:,1])
    
    hr_loss = loss_heating_rate(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred, delta_pressure)
    
    return hr_loss

def loss_energy_wrapper(data, y_pred):
    _, _, _, delta_pressure, y_true = data

    (flux_down_pred, flux_up_pred, _, _) = y_pred
    (flux_down_true, flux_up_true) = (y_true[:,:,0], y_true[:,:,1])
    
    loss = loss_energy(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred)
    
    return loss


def loss_flux(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred):  

    flux_pred = torch.concat((flux_down_pred,flux_up_pred),dim=1)
    flux_true = torch.concat((flux_down_true,flux_up_true),dim=1)

    flux_loss = torch.sqrt(torch.mean(torch.square(flux_pred - flux_true), 
                       dim=(0,1), keepdim=False))

    return flux_loss

def loss_avg_flux(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred):  

    flux_pred = torch.concat((flux_down_pred,flux_up_pred),dim=1)
    flux_true = torch.concat((flux_down_true,flux_up_true),dim=1)

    flux_loss = torch.mean(flux_pred - flux_true, 
                       dim=(0,1), keepdim=False)

    return flux_loss

def loss_full_flux_wrapper(data, y_pred):
    _, _, _, _, y_true = data
    (flux_down_pred, flux_up_pred, _, _) = y_pred
    (flux_down_true, flux_up_true) = (y_true[:,:,0], y_true[:,:,1])
    loss = loss_flux(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred)
    return loss

def loss_avg_flux_wrapper(data, y_pred):
    _, _, _, _, y_true = data
    (flux_down_pred, flux_up_pred, _, _) = y_pred
    (flux_down_true, flux_up_true) = (y_true[:,:,0], y_true[:,:,1])
    loss = loss_avg_flux(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred)
    return loss

def loss_clear_flux_wrapper(data, y_pred):
    _, _, _, _, y_true = data
    (_, _, flux_down_pred, flux_up_pred) = y_pred
    (flux_down_true, flux_up_true) = (y_true[:,:,2], y_true[:,:,3])
    loss = loss_flux(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred)
    return loss

def loss_henry_wrapper(data, y_pred):
    loss_full_flux = loss_full_flux_wrapper(data, y_pred)
    loss_clear_flux = loss_clear_flux_wrapper(data, y_pred)
    loss_full_heating_rate = loss_full_heating_rate_wrapper(data, y_pred)
    loss_clear_heating_rate = loss_clear_heating_rate_wrapper(data, y_pred)

    w1 = 2.0
    w2 = 1.0
    w3 = 1.0
    w4 = 0.5

    loss = (1.0 / (w1 + w2 + w3 + w4)) * (w1 * loss_full_flux + w2 * loss_clear_flux + w3 * loss_full_heating_rate + w4 * loss_clear_heating_rate)

    return loss
def loss_henry_wrapper_2(data, y_pred):
    loss_full_flux = loss_full_flux_wrapper(data, y_pred)
    loss_clear_flux = loss_clear_flux_wrapper(data, y_pred)


    w1 = 2.0
    w2 = 1.0
    w3 = 1.0
    w4 = 0.5

    loss = (1.0 / (w1 + w2)) * (w1 * loss_full_flux + w2 * loss_clear_flux)

    return loss

def train_loop(dataloader, model, optimizer, loss_function, device):
    """ Generic training loop """

    torch.cuda.synchronize()
    t_1 = time.time()
    model.train()

    loss_string = "Training Loss: "
    for batch, data in enumerate(dataloader):
        data = [x.to(device) for x in data]
        y_pred = model(data)
        torch.cuda.synchronize()
        t_0 = time.time()
        loss = loss_function(data, y_pred)
        torch.cuda.synchronize()
        t_01 = time.time()
        global t_loss
        t_loss += t_01 - t_0

        if False:
            with torch.autograd.profiler.profile(use_cuda=True,
                                             use_cpu=True,
                                             with_modules=True,
                                             with_stack=True) as prof:
                loss.backward()
            print(prof.key_averages(group_by_stack_n=8).table(sort_by="cuda_time_total"))
        else:
            loss.backward()
        torch.cuda.synchronize()
        t_03 = time.time()
        global t_backward
        t_backward += t_03 - t_01
        optimizer.step()
        optimizer.zero_grad()

        if batch % 20 == 0:
            loss_value = loss.item()
            loss_string += f" {loss_value:.9f}"
        torch.cuda.synchronize()
        t_02 = time.time()
        global t_grad
        t_grad += t_02 - t_01
    #print (loss_string)
    torch.cuda.synchronize()
    t_2 = time.time()
    global t_train
    t_train += t_2 - t_1

def test_loop(dataloader, model, loss_functions, loss_names, device):
    """ Generic testing / evaluation loop """
    model.eval()
    num_batches = len(dataloader)

    loss = np.zeros(len(loss_functions), dtype=np.float32)

    with torch.no_grad():
        for data in dataloader:
            data = [x.to(device) for x in data]
            y_pred = model(data)
            for i, loss_fn in enumerate(loss_functions):
                loss[i] += loss_fn(data, y_pred).item()

    loss /= num_batches

    print(f"Test Error: ")
    for i, value in enumerate(loss):
        print(f" {loss_names[i]}: {value:.8f}")
    print("")

    return loss

def test_loop_internals (dataloader, model, loss_functions, loss_names, device):
    """ Generic testing / evaluation loop """
    model.eval()
    num_batches = len(dataloader)

    loss = np.zeros(len(loss_functions), dtype=np.float32)

    lwp = []
    iwp = []
    o3 = []
    mu_diffuse = []
    s_direct   = []
    s_diffuse   = []
    r_toa = []
    r_surface = []
    mu_direct = []
    t_direct = []
    t_diffuse = []
    h2o = []

    with torch.no_grad():
        for data in dataloader:
            data = [x.to(device) for x in data]
            y_pred, internal_data = model(data)
            lwp.append(internal_data[0])
            iwp.append(internal_data[1])
            o3.append(internal_data[2])
            mu_diffuse.append(internal_data[3])
            s_direct.append(internal_data[4])
            s_diffuse.append(internal_data[5])
            r_toa.append(internal_data[6])
            r_surface.append(internal_data[7])
            mu_direct.append(internal_data[8])
            t_direct.append(internal_data[9])
            t_diffuse.append(internal_data[10])
            h2o.append(internal_data[11])
            for i, loss_fn in enumerate(loss_functions):
                loss[i] += loss_fn(data, y_pred).item()

    loss /= num_batches

    print(f"Test Error: ")
    for i, value in enumerate(loss):
        print(f" {loss_names[i]}: {value:.8f}")
    print("")

    lwp = torch.cat(lwp, dim=0)
    iwp = torch.cat(iwp, dim=0)
    o3 = torch.cat(o3, dim=0)
    mu_diffuse = torch.cat(mu_diffuse, dim=0)
    mu_direct = torch.cat(mu_direct, dim=0)
    s_direct = torch.cat(s_direct, dim=0)
    s_diffuse = torch.cat(s_diffuse, dim=0)
    r_toa = torch.cat(r_toa, dim=0)
    r_surface = torch.cat(r_surface, dim=0)
    t_direct = torch.cat(t_direct, dim=0)
    t_diffuse = torch.cat(t_diffuse, dim=0)
    h2o = torch.cat(h2o, dim=0)

    internal_data = [lwp, iwp, o3, mu_diffuse, s_direct, s_diffuse, r_toa, r_surface, mu_direct, t_direct, t_diffuse, h2o]

    return loss, internal_data

def tensorize(np_ndarray):
    t = torch.from_numpy(np_ndarray).float()
    return t


def train_full_dataloader():

    print("Pytorch version:", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    if torch.cuda.is_available():
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:',torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
        print(f'Device capability = {torch.cuda.get_device_capability()}')
        use_cuda = True
    else:
        use_cuda = False

    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32 = True
    #fast_train_loop = torch.compile(train_loop, mode="reduce-overhead")

    datadir     = "/data-T1/hws/tmp/"
    mode = "training"
    #mode = "testing"
    year = "2008"
    #year = "2015"
    train_input_dir = f"/data-T1/hws/CAMS/processed_data/{mode}/{year}/"
    cross_input_dir = "/data-T1/hws/CAMS/processed_data/cross_validation/2008/"
    months = [str(m).zfill(2) for m in range(1,13)]
    train_input_files = [f'{train_input_dir}nn_input-{mode}-{year}-{month}.nc' for month in months]
    cross_input_files = [f'{cross_input_dir}nn_input-cross_validation-2008-{month}.nc' for month in months]

    #batch_size = 2048
    #batch_size = 1536 
    batch_size = 1024
    n_channel = 48
    n_constituent = 9
    n_band = 16
    

    filename_full_model = datadir + f"/Torch.LW.v11." # scattering_v2_efficient

    is_initial_condition = False
    if is_initial_condition:
        checkpoint_period = 1
        epochs = 1
        t_start = 0
        number_of_tries = 20
    else:
        checkpoint_period = 5 
        epochs = 2000 

        number_of_tries = 1
        

        if False:
            #initial_model_n = 0   #v4
            initial_model_n = 0   #v6, v7
            t_start = 1
            filename_full_model_input = f'{filename_full_model}i' + str(initial_model_n).zfill(2)
        else:
            t_start = 65 #0
            filename_full_model_input = filename_full_model + str(t_start).zfill(3)


    for ee in range(number_of_tries):

        #print(f"Model = {str(ee).zfill(3)}")


        #filename_full_model = filename_full_model_input  #

        t_warmup = 1  # for profiling
        t = t_start
        dropout_p = 0.00

        #dropout_schedule = (0.0, 0.07, 0.1, 0.15, 0.2, 0.15, 0.1, 0.07, 0.0, 0.0) 

        #dropout_epochs =   (-1, 40, 60, 70,  80, 90,  105, 120, 135, epochs + 1)


        # Used for v11
        dropout_schedule = (0.0, 0.07, 0.15, 0.07, 0.0, 0.0) 
        # Used for v11
        dropout_epochs =   (-1, 20, 23, 27, 35, epochs + 1)

        # 400
        #dropout_epochs =   (-1, 200,   300, 350,  400, 450,  550, 650, 750, epochs + 1)
        #dropout_epochs =   (-1, 40,   60, 70,  80, 90,  110, 130, 150, epochs + 1) #v6


        #v7-v15
        #dropout_epochs =   (-1, 65, 85, 95,  105, 115,  135, 140, 145, epochs + 1) 

        # changed 65 to 40 for v17
        #dropout_epochs =   (-1, 40, 85, 95,  105, 115,  120, 125, 130, epochs + 1)

        # changes for v18, v19 - v25, v27, v29
        #dropout_epochs =   (-1, 40, 60, 70,  80, 90,  95, 100, 105, epochs + 1)

        # v29

        #V26
        #dropout_epochs =   (-1, 40, 60, 70,  105, 140,  150, 160, 170, epochs + 1)



        # v28
        #dropout_epochs =   (-1, 80, 100, 130,  150, 165,  175, 185, 195, epochs + 1) 

        dropout_index = next(i for i, x in enumerate(dropout_epochs) if t <= x) - 1
        dropout_p = dropout_schedule[dropout_index]
        last_dropout_index = dropout_index

        model = FullNet(n_channel,n_constituent,n_band,dropout_p,device).to(device=device)

        n_parameters = count_parameters(model)

        print(f"Number of parameters = {n_parameters}")

        if is_initial_condition:
                n_parameters = count_parameters(model)

                t2 = count_parameters(model.extinction_net)
                print(f"Number of extinction = {t2}")

                t3 = count_parameters(model.scattering_net)
                print(f"Number of scattering = {t3}")

                if False:
                    t4 = count_parameters(model.scattering_net.direct_scattering)
                    print(f"Number of scattering - direct scattering = {t4}")

                    t5 = count_parameters(model.scattering_net.direct_selection)
                    print(f"Number of scattering - direct selection = {t5}")

        # v7 lr increased from 0.001 to 0.0025

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_dataset = RT_lw_data.RTDataSet(train_input_files)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, 
                                                    shuffle=False,
                                                            num_workers=1)
        
        validation_dataset = RT_lw_data.RTDataSet(cross_input_files)

        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size, 
                                                    shuffle=False,
                                                            num_workers=1)
        
        #start = torch.cuda.Event(enable_timing=True)
        #end = torch.cuda.Event(enable_timing=True)

        loss_functions = (loss_henry_wrapper, loss_full_flux_wrapper, loss_clear_flux_wrapper, loss_full_heating_rate_wrapper, loss_clear_heating_rate_wrapper)
        loss_names = ("Loss", "Full Flux Loss", "Clear Flux Loss","Full Heating Rate Loss","Clear Heating Rate Loss")
        
        #loss_functions = (loss_flux_wrapper)
        #loss_names = ("Flux Loss")

        if t > 0:
            checkpoint = torch.load(filename_full_model_input)
            print(f"Loaded Model: epoch = {filename_full_model_input}")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #epoch = checkpoint['epoch']

        print(f"       dropout = {dropout_p}")
        while t < epochs:
            t += 1

            dropout_index = next(i for i, x in enumerate(dropout_epochs) if t <= x) - 1
            if dropout_index != last_dropout_index:
                last_dropout_index = dropout_index
                dropout_p = dropout_schedule[dropout_index]
                model.reset_dropout(dropout_p)
            print(f"Epoch {t}\n-------------------------------")
            print(f"Dropout: {dropout_p}")

            if True:


                #with profile(
                #    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                #    with_stack=True, with_modules=True,
                #) as prof:
                #start.record() loss_flux_full_wrapper
                train_loop(train_dataloader, model, optimizer, 
                           loss_henry_wrapper, 
                           device)
                
                if False:
                    print(f"Total Train time= {t_train}")
                    print(f"Full time forward total = {t_total}")
                    print(f"Full time loss = {t_loss}")    
                    print(f"Full time grad = {t_grad}")  
                    print(f"Full time backward = {t_backward}")  
                    print(f"Full time extinction = {t_extinction}")
                    print(f"Full time scattering = {t_scattering_v2_tau}")
                    print(f"Time for scattering = {t_direct_scattering}")
                    print(f"Time for split = {t_direct_split}")

                loss = test_loop(validation_dataloader, model, loss_functions, loss_names, device)

                #if use_cuda: 
                    #torch.cuda.synchronize()
                #end.record()

                #print(f"\n Elapsed time in seconds: {start.elapsed_time(end) / 1000.0}\n")

                #print(prof.key_averages(group_by_stack_n=6).table(sort_by='self_cpu_time_total', row_limit=15))
            if t % checkpoint_period == 0:
                if is_initial_condition:
                    torch.save({
                    'epoch': t,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, filename_full_model + 'i' + str(ee).zfill(2))
                    print(f' Wrote Initial Model: {ee}')
                else:
                    torch.save({
                    'epoch': t,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, filename_full_model + str(t).zfill(3))
                    print(f' Wrote Model: epoch = {t}')

        print("Done!")



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def write_internal_data(internal_data, output_file_name):
    import xarray as xr
    lwp, iwp, o3, mu_diffuse, s_direct, s_diffuse, r_toa, r_surface, mu_direct, t_direct, t_diffuse, h2o = internal_data

    shape = lwp.shape
    shape2 = lwp.numpy().shape

    example = np.arange(shape[0])
    layer = np.arange(shape[1])

    #lwp = xr.DataArray(lwp, coords=[time,site,layer], dims=("time","site","layer"), name="lwp")

    #iwp = xr.DataArray(iwp, coords=[time,site,layer], dims=("time","site","layer"), name="iwp")

    #r = xr.DataArray(r, coords=[time,site,layer],dims=("time","site","layer"), name="r")

    mu_diffuse = mu_diffuse.numpy().flatten()
    mu_direct = mu_direct.numpy()
    #s1 = np.shape(mu_direct)
    #mu_direct = np.reshape(mu_direct, (s1[0], s1[1]*s1[2]))

    rs_direct = s_direct.numpy()
    rs_diffuse = s_diffuse.numpy()
    rr_toa = r_toa.numpy()
    rr_surface = r_surface.numpy()

    is_bad = np.isnan(rs_direct).any() or np.isnan(rs_diffuse).any()
    print(f"is bad = {is_bad}")

    ds = xr.Dataset(
        data_vars = {
            "lwp": (["example","layer"], lwp.numpy()),
            "iwp": (["example","layer"], iwp.numpy()),
            "o3": (["example","layer"], o3.numpy()),
            "mu_diffuse"  : (["example"], mu_diffuse),
            "mu_direct"  : (["example"], mu_direct[:,0,0]),
            "s_direct": (["example","layer"], rs_direct),
            "s_diffuse": (["example","layer"], rs_diffuse),
            "r_toa" : (["example"], rr_toa),
            "r_surface" : (["example"], rr_surface),
            "t_direct": (["example","layer"], t_direct.numpy()),
            "t_diffuse": (["example","layer"], t_diffuse.numpy()),
            "h2o": (["example","layer"], h2o.numpy()),
            },
         coords = {
             "example" : example,
             "layer" : layer,
         },
    )

    ds.to_netcdf(output_file_name)
    ds.close()


def test_full_dataloader():

    print("Pytorch version:", torch.__version__)
    device = "cpu"
    print(f"Using {device} device")

    datadir     = "/data-T1/hws/tmp/"
    batch_size = 1024
    n_channel = 48
    n_constituent = 9
    n_band = 16
    is_use_internals = False #True

    if is_use_internals:
        model = FullNetInternals(n_channel,n_constituent,n_band,dropout_p=0,device=device)
    else:
        model = FullNet(n_channel,n_constituent,n_band,dropout_p=0,device=device)

    #print(model)

    if False:
        n_parameters = count_parameters(model)

        t1 = count_parameters(model.mu_diffuse_net) + count_parameters(model.spectral_net)
        print(f"Number of diffuse, spectral = {t1}")

        t2 = count_parameters(model.extinction_net)
        print(f"Number of extinction = {t2}")

        t3 = count_parameters(model.scattering_net)
        print(f"Number of scattering = {t3}")


    model = model.to(device=device)

    version_name = 'v11'
    filename_full_model = datadir + f"/Torch.LW.{version_name}." 

    years = ("2009", "2015", "2020")
    mode = "testing"
    #years = ("2020", )

    for year in years:
        test_input_dir = f"/data-T1/hws/CAMS/processed_data/testing/{year}/"
        months = [str(m).zfill(2) for m in range(1,13)]
        test_input_files = [f'{test_input_dir}nn_input-{mode}-{year}-{month}.nc' for month in months]
        #test_input_files = ["/data-T1/hws/tmp/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.2.nc"]


        test_dataset = RT_lw_data.RTDataSet(test_input_files)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, 
                                                    shuffle=False,
                                                            num_workers=1)

        
        loss_functions = (loss_henry_wrapper, loss_full_flux_wrapper, 
        loss_avg_flux_wrapper, loss_energy_wrapper, loss_clear_flux_wrapper, loss_full_heating_rate_wrapper, loss_clear_heating_rate_wrapper)
        loss_names = ("Loss", "Full Flux Loss", "Avg Flux Loss", "Avg Loss Energy","Clear Flux Loss","Full Heating Rate Loss","Clear Heating Rate Loss")

        print(f"Testing error, Year = {year}")
        for t in range(40, 60,5):

            checkpoint = torch.load(filename_full_model + str(t).zfill(3), map_location=torch.device(device))
            print(f"Loaded Model: epoch = {t}")
            model.load_state_dict(checkpoint['model_state_dict'])

            #print(f"Total number of parameters = {n_parameters}", flush=True)
            #print(f"Spectral decomposition weights = {model.spectral_net.weight}", flush=True)

            if is_use_internals:
                loss, internal_data = test_loop_internals (test_dataloader, model, loss_functions, loss_names, device)
                write_internal_data(internal_data, output_file_name=test_input_dir + f"internal_output.sc_{version_name}_{t}.{year}.nc")
            else:
                loss = test_loop (test_dataloader, model, loss_functions, loss_names, device)
    

if __name__ == "__main__":
    #train_direct_only()
    #train_full()
    #
    #test_full()

    #global t_direct_scattering = 0.0
    #global t_direct_split = 0.0
    #t_scattering_v2_tau = 0.0
    
    #train_full_dataloader()
    test_full_dataloader()

   