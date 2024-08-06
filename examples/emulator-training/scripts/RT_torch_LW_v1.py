# Same as RT_torch_12 except mass extinction depends only on (T,P) 
# instead of (T,P,ln(P))
# Added variants for cost function, cost_henry_2 with various weightings
# of flux and heating rate and greater weight for direct flux, direct heat

# Same as RT_torch_12_v2 except
# Except more channels

# Same as RT_torch_12_v3 except
# scattering model using m different scattering models, where m < n_channels

# Same as RT_torch_12_v4 except
# loss separates direct and diffuse components
# also eliminated some unused functions

# Same as RT_torch_12.v5 (skips RT_torch_12.v6) except uses drop out 
# on extinction "effective mass" weights

# Same as RT_torch_12.v5 (skips RT_torch_12.v6, v7) except uses exp
# on inputs to scattering

import numpy as np
import time
from typing import List
import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F

from RT_data_hws import load_data_direct_pytorch, load_data_full_pytorch_2, absorbed_flux_to_heating_rate
import RT_data_hws_2

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
        
        filter_h2o = torch.stack([filter_h2o, filter_h2o, filter_h2o])

        filter_o3 = torch.tensor([0,0,0,1,1, 0,1,1,0,0, 0,0,1,0,0, 0],
                                       dtype=torch.float32,device=device)

        filter_o3 = torch.stack([filter_o3, filter_o3, filter_o3])

        filter_co2 = torch.tensor([0,0,1,1,1, 1,1,1,0,0, 0,1,1,1,1, 0],
                                       dtype=torch.float32,device=device)
        
        filter_co2 = torch.stack([filter_co2, filter_co2, filter_co2])
        
        # used for both o2 and n2 since they are both constants
        filter_o2  = torch.tensor([1,0,0,0,0, 0,0,0,0,0, 1,0,0,0,0, 0],
                                       dtype=torch.float32,device=device)
        
        filter_o2 = torch.stack([filter_o2, filter_o2, filter_o2])

        filter_n2o = torch.tensor([0,0,1,0,0, 0,0,1,1,0, 0,0,1,0,1, 0],
                                       dtype=torch.float32,device=device)

        filter_n2o = torch.stack([filter_n2o, filter_n2o, filter_n2o])
        
        filter_ch4 = torch.tensor([0,0,0,0,0, 0,0,0,1,0, 0,0,0,0,0, 1,],
                                       dtype=torch.float32,device=device)
        
        filter_ch4 = torch.stack([filter_ch4, filter_ch4, filter_ch4])

        filter_co = torch.tensor([0,0,0,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,],
                                       dtype=torch.float32,device=device)
        
        filter_co = torch.stack([filter_co, filter_co, filter_co])

        self.filter_h2o = filter_h2o.transpose(1,0).flatten()

        self.filter_o3 = filter_o3.transpose(1,0).flatten()

        #print(f'filter_o3 = {self.filter_o3}')

        self.filter_co2 = filter_co2.transpose(1,0).flatten()
        
        self.filter_o2  = filter_o2.transpose(1,0).flatten()

        self.filter_n2o = filter_n2o.transpose(1,0).flatten()
        
        self.filter_ch4 = filter_ch4.transpose(1,0).flatten()

        self.filter_co = filter_co.transpose(1,0).flatten()

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
        t_p = temperature_pressure

        #def d(value):
        #    return F.dropout(value,p=self.dropout_p,training=self.training)
        
        a = torch.exp
        b = torch.sigmoid

        one = torch.ones((c.shape[0],1),dtype=torch.float32,device=self.device)
        tau_lw  = a(self.net_lw (one)) * (c[:,0:1])
        tau_iw  = a(self.net_iw (one)) * (c[:,1:2])
        tau_h2o = a(self.net_h2o(one)) * (c[:,2:3]) * (self.filter_h2o * 
                                                       b(self.net_ke_h2o(t_p)))
        tau_o3  = a(self.net_o3 (one)) * (c[:,3:4] * (self.filter_o3  * 
                                                       b(self.net_ke_o3 (t_p))))
        tau_co2 = a(self.net_co2(one)) * (c[:,4:5]) * (self.filter_co2 * 
                                                       b(self.net_ke_co2(t_p)))
        tau_o2   = a(self.net_u  (one)) * (c[:,5:6]) * (self.filter_o2 * 
                                                       b(self.net_ke_o2  (t_p)))
        tau_n2o = a(self.net_n2o(one)) * (c[:,6:7]) * (self.filter_n2o * 
                                                       b(self.net_ke_n2o(t_p)))
        tau_ch4 = a(self.net_ch4(one)) * (c[:,7:8]) * (self.filter_ch4 * 
                                                       b(self.net_ke_ch4(t_p)))

        tau_co = a(self.net_co(one)) * (c[:,8:9]) * (self.filter_co * 
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

        #n_hidden = [7, 7, 7, 7]  # settings for 8


        #n_hidden = [24, 24, 24, 24] # settings for 12
        n_hidden = [32, 32, 32] # settings for 13

        # Create basis functions for scattering

        # Has additional input for zenith angle ('mu_direct')
        # Added bias=True for v14


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
        tau = x

        #print(f"tau.shape = {tau.shape}")
        # sum over constituents
        tau_total = torch.sum(tau, dim=2, keepdims=False)


        t_diffuse = torch.exp(-tau_total)



        # f = number of features
        # [i,channels,f]


        e_split_diffuse = self.diffuse_scattering(tau)
        n = e_split_diffuse.shape[0]
        
        # [i,channels,3, m]


        e_split_diffuse = torch.reshape(e_split_diffuse,
                                       (n, self.n_channel,
                                        self.n_scattering_nets, 3))
        # [i,channels, m, 3]
        e_split_diffuse = F.softmax(e_split_diffuse,dim=-1)

        e_split_diffuse = self.diffuse_selection(e_split_diffuse)

        # [i, channels, 1, 3]

        e_split_diffuse = torch.squeeze(e_split_diffuse, dim=-2)

        # [i,channels,3]  

        e_split_diffuse = F.softmax(e_split_diffuse, dim=-1)


        layers = [t_diffuse, e_split_diffuse]

        return layers



class MultiReflection(nn.Module):
    """ 
    Recomputes each layer's radiative coefficients by accounting
    for interaction (multireflection) with all other layers using the 
    Adding-Doubling method (no learning).
    """

    def __init__(self, n_channels, n_bands, device):
        super(MultiReflection, self).__init__()
   


        weight_values = torch.rand((n_channels, n_bands),
                                   requires_grad=True,device=device,
                                   dtype=torch.float32,)
        
        self.bands_to_channels = nn.parameter.Parameter(weight_values, requires_grad=True)

    def _compute_surface_reflection(self,r,t):
        """

        Bottom up computation of cumulative surface reflection
        and denominator factor

        n = len(r)
        Output has same size

        rs index corresponds to newly formed virtual surface 
        d index corresponds to virtual surface (layer that flow is entering)

        """

        # n = len(r)
        # Start at bottom of the atmosphere (layer = n-1)
        rs = []
        last_rs = r[:,-1,:]  # reflectance of original surface is unchanged

        rs.append(last_rs)
        ds = []
        ds.append(torch.ones())

        # n-2 . . 0 (inclusive)
        for l in reversed(torch.arange(start=0, end=r.shape[1]-1)):
            dd = 1.0 / (1.0 - last_rs * r[:,l,:])
            ds.append(dd)
            last_rs = r[:,l,:] + last_rs * t[:,l,:] * t[:,l,:] * dd
            rs.append(last_rs)

        rs = torch.stack(rs,dim=1)  
        ds = torch.stack(ds,dim=1)  

        rs = torch.flip(rs, dims=(1,))  # n values: 0 .. n-1 (includes surface)
        ds = torch.flip(ds, dims=(1,)) # n values: 0 .. n-1 (last value is one)

        return rs, ds
    
    def _compute_top_reflection(self,r,t):
        """
        In top down order compute cumulative upper layer reflection
        and denominator factor

        rt index corresponds to newly formed virtual surface 
        d is aligned with the value of rt it computed, corresponds
        to layer it is flowing into
        """

        # Start at top of the atmosphere, first layer in isolation
        rt = []
        last_rt = r[:,0,:]  # reflectance of top surface is unchanged
        rt.append(last_rt)
        dt = []
        dt.append(torch.ones())

        for l in torch.arange(start=1, end=r.shape[1]):
            dd = 1.0 / (1.0 - last_rt * r[:,l,:])
            dt.append(dd)
            last_rt = r[:,l,:] + last_rt * t[:,l,:] * t[:,l,:] * dd
            rt.append(last_rt)

        rt = torch.stack(rt,dim=1)  # n values 
        dt = torch.stack(dt,dim=1)  

        return rt, dt

    def _compute_sandwich_d(self,rs,rt):
        """

        """
        # Start at top of the atmosphere 

        d = []
        d.append(torch.ones())

        # Pad at the beginning????

        for l in torch.arange(start=1, end=rs.shape):
            dd = 1.0 / (1.0 - rt[l-1] * rs[l])
            d.append(dd)

        d = torch.stack(d,dim=1)  # n values

        return d
    
    def _compute_upward_flux (self, s_up, rt, dt, t, a):
        """ 
        Input:
            s_up: n elements, index represent layer flux is exiting
            
        Output:
            flux_up: n elements from layer
            absorbed_flux: n elements: into surface + layers (surface is zero since going up)
        """
        flux = torch.zeros()
        flux_up = []
        absorbed_flux = []
        absorbed_flux.append(torch.zeros()) # no absorbed flux, last layer = n-1

        # from n to 1
        for l in reversed(torch.arange(1,s_up.shape[1]-1)):
            flux += s_up[:,l+1,:] # initial exits layer n-1
            flux_up.append(flux) 
            # initially absorbed at layer n-2
            a_multi = a[:,l,:] * (1.0 + t[:,l,:] * rt[:,l-1,:] * dt[:,l,:])
            absorbed_flux.append(flux * a_multi) # initial absorbed at l-1 or n-2
            # propagate into next layer
            flux = flux * t[:,l,:] * dt[:,l,:]
        flux += s_up[:,1,:]
        flux_up.append(flux)
        absorbed_flux.append(flux * a[:,0,:])
        flux = flux * t[:,0,:]

        flux += s_up[:,0,:] # from layer zero toward upper atmosphere
        flux_up.append(flux)

        flux_up = torch.stack(flux_up, dim=1) # from layer up, n values
        flux_up = torch.flip(flux_up, dims=(1,))

        absorbed_flux = torch.stack(absorbed_flux, dim=1) # n values
        absorbed_flux = torch.flip(absorbed_flux, dims=(1,))

        return absorbed_flux, flux_up
    
    def _compute_downward_flux (self, s_down, rs, ds, t, a):
        """ 
        s_down represents layer it is exiting
        n-1 = len(s_down)

        index of flux represents radiation flowing into layer
        n = len(output)
        """

        # no downward flux or absorption for layer=0
        flux = torch.zeros()
        flux_down = []
        flux_down.append(flux)
        absorbed_flux = []
        absorbed_flux.append(torch.zeros())

        for l in torch.arange(0, s_down.shape[1]-1):
            flux += s_down[:,l,:]  # exiting layer l, entering l+1
            flux_down.append(flux)
            a_multi = a[:,l+1,:] * (1 + rs[:,l+2,:] * t[:,l+1,:] * ds[:,l+1,:]) #good
            absorbed_flux.append(a_multi * flux)
            flux = flux * t[:,l+1,:] * ds[:,l+1,:]

        flux += s_down[:,-1,:]
        flux_down.append(flux)
        absorbed_flux.append(a[:,-1,:] * flux)

        absorbed_flux = torch.stack(absorbed_flux, dim=1)
        flux_down = torch.stack(flux_down, dim=1)

        # n output values
        return absorbed_flux, flux_down

    def _adding_doubling (self, a, r, t, s):
        """
        All inputs:
            Dimensions[examples, layers, channels]
        """

        # Bottom up cumulative surface reflection
        rs, ds = self._compute_surface_reflection(r,t)

        # Top down cumulative top layer reflection
        rt, dt = self._compute_top_reflection(r,t)

        # compute multi-reflection sandwich terms 
        # uses rt for top, rs for bottom

        d_multi = self._compute_sandwich_d(rs, rt)

        ### Downward sources
        s_multi_down = s[:,:-1,:] * d_multi[:,1:,:]  # n-1 values, flow from layer
        s_multi_up_down = s[:,1:,:] * rt[:,:-1,:] * d_multi[:,1:,:]

        s_down = s_multi_down + s_multi_up_down # index corresponds to exiting layer

        absorbed_flux_down, flux_down = self._compute_downward_flux (s_down, rs, ds, t, a)

        ### Upward sources
        s_multi_up = s * d_multi    # Index of exiting surface, n values
        s_multi_down_up = s[:,:-1,:] * rs[:,1:,:] * d_multi[:,1:,:] # n-1 values, index of entering surface
        s_multi_down_up = torch.cat([torch.zeros(), s_multi_down_up], axis=1) #n values, index of exiting surface
 
        s_up = s_multi_up + s_multi_down_up

        absorbed_flux_up, flux_up = self._compute_upward_flux (s_up, rt, dt, t, a)

        absorbed_flux = absorbed_flux_down + absorbed_flux_up

        return flux_down, flux_up, absorbed_flux
    
    def forward(self, x):
        """
        Computations are independent across channel.

        The prefixes -- t, e, r, a -- correspond respectively to
        transmission, extinction, reflection, and absorption.
        """

        raw_sources, layers, x_surface = x

        t_diffuse, e_split_diffuse = layers

        shape = t_diffuse.shape

        (r_surface_diffuse, a_surface_diffuse) = (x_surface[:,:,0:1], 
                                               x_surface[:,:,1:2])

        # (n_examples, n_levels, n_bands) * (n_bands, n_channels)
        s = torch.matmul(raw_sources, F.softmax(self.bands_to_channels,axis=1)) ##???

        e = 1.0 - t_diffuse

        t = t_diffuse + e * e_split_diffuse[:,:,:,0]
        r = e * e_split_diffuse[:,:,:,1]
        a = e * e_split_diffuse[:,:,:,2]

        r_surface_diffuse = r_surface_diffuse.expand(shape[0],1,shape[2])
        a_surface_diffuse = a_surface_diffuse.expand(shape[0],1,shape[2])
        t_surface_diffuse = torch.zeros((shape[0],1,shape[2]))

        a = torch.stack((a, a_surface_diffuse), dim=1)
        r = torch.stack((r, r_surface_diffuse), dim=1)
        t = torch.stack((t, t_surface_diffuse), dim=1)

        return _adding_doubling(a, r, t, s)
 
    
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

        x_layers, x_surface, raw_sources, _, _, _, _ = x

        #print(f"x_layers.shape = {x_layers.shape}")
        #9 constituents: lwc, ciw, h2o, o3, co2,  o2, n2o, ch4, co,  -no2?, 
        (temperature_pressure, 
        constituents) = (x_layers[:,:,0:2], 
                        x_layers[:,:,2:11])
    
        tau = self.extinction_net((temperature_pressure, 
                                constituents))

        layers = self.scattering_net((tau,))

        flux = self.multireflection_net([raw_sources, layers, x_surface])
        return flux


def loss_weighted(y_true, y_pred, weight_profile):
    error = torch.sqrt(torch.mean(torch.square(weight_profile * (y_pred - y_true)), 
                       dim=(0,1), keepdim=False))
    return error

def loss_heating_rate_direct(y_true, y_pred, toa, delta_pressure):
    absorbed_true = y_true[:,:-1] - y_true[:,1:]
    absorbed_pred = y_pred[:,:-1] - y_pred[:,1:]
    heat_true = absorbed_flux_to_heating_rate(absorbed_true, delta_pressure)
    heat_pred = absorbed_flux_to_heating_rate(absorbed_pred, delta_pressure)
    error = torch.sqrt(torch.mean(torch.square(toa * (heat_true - heat_pred)),
                                  dim=(0,1),keepdim=False))
    return error

def loss_heating_rate_direct_wrapper(data, y_pred, weight_profile):
    _, y, x_toa, x_delta_pressure = data
    loss = loss_heating_rate_direct(y,y_pred,x_toa,x_delta_pressure)
    return loss

def loss_heating_rate_direct_full_wrapper(data, y_pred):
    _, _, toa, delta_pressure, y_true, _ = data
    flux_down_direct_pred, _, _, _ = y_pred
    flux_down_direct_true = y_true[:,:,0]
    loss = loss_heating_rate_direct(flux_down_direct_true, 
                                    flux_down_direct_pred, 
                                    toa, 
                                    delta_pressure)
    return loss

def loss_heating_rate_full(flux_absorbed_true, flux_absorbed_pred, 
                           toa, delta_pressure):
    heat_true = absorbed_flux_to_heating_rate(flux_absorbed_true, 
                                              delta_pressure)
    heat_pred = absorbed_flux_to_heating_rate(flux_absorbed_pred, 
                                              delta_pressure)
    loss = torch.sqrt(torch.mean(torch.square(toa * (heat_true - heat_pred)),
                                  dim=(0,1),keepdim=False))
    return loss

def loss_heating_rate_full_wrapper(data, y_pred):
    _, _, toa, delta_pressure, _, flux_absorbed_true = data
    _, _, _, flux_absorbed_pred = y_pred

    loss = loss_heating_rate_full(flux_absorbed_true, flux_absorbed_pred, 
                                  toa, delta_pressure)
    return loss


def loss_flux_direct_wrapper(data, y_pred, weight_profile):
    _, y_true, _, _ = data
    loss = loss_weighted(y_true,y_pred,weight_profile)
    return loss

def loss_flux_full_wrapper(data, y_pred):
    _, _, toa, _, y_true, _ = data
    flux_down_true, flux_up_true = y_true[:,:,1], y_true[:,:,2]

    (flux_down_direct_pred, flux_down_diffuse_pred, 
     flux_up_diffuse_pred, _) = y_pred

    flux_down_pred = flux_down_direct_pred + flux_down_diffuse_pred
    flux_up_pred = flux_up_diffuse_pred
    
    flux_pred = torch.concat((flux_down_pred,flux_up_pred),dim=1)
    flux_true = torch.concat((flux_down_true,flux_up_true),dim=1)
    loss = loss_weighted(flux_true, flux_pred, toa)
    return loss

def loss_heat_flux_full_wrapper(data, y_pred):
    weight_heat = 0.3
    loss_flux = loss_flux_full_wrapper(data, y_pred)
    loss_heat = loss_heating_rate_full_wrapper(data, y_pred)
    loss = weight_heat * loss_heat + (1.0 - weight_heat) * loss_flux

    return loss

def loss_flux_direct_wrapper_2(data, y_pred):
    _, _, toa, _, y_true, _ = data
    flux_down_direct_pred, _, _, _ = y_pred
    flux_down_direct_true = y_true[:,:,0]
    
    loss = loss_weighted(flux_down_direct_true, flux_down_direct_pred, toa)
    return loss

def loss_heating_rate_diffuse(y_true, y_pred, toa_weighting_profile, 
               delta_pressure):
    # Handles flux using TOA weight rather than weight profile
    
    (flux_down_direct_pred, flux_down_diffuse_pred, 
     flux_up_diffuse_pred, flux_absorbed_pred) = y_pred
    
    (flux_down_direct_true, flux_down_true, 
     flux_up_diffuse_true) = (y_true[:,:,0], y_true[:,:,1], y_true[:,:,2])
    flux_down_diffuse_true = flux_down_true - flux_down_direct_true


    flux_absorbed_diffuse_true = (flux_down_diffuse_true[:,:-1] -
                             flux_down_diffuse_true[:,1:] + 
                             flux_up_diffuse_true[:,1:] -
                             flux_up_diffuse_true[:,:-1])

    flux_absorbed_diffuse_pred = (flux_down_diffuse_pred[:,:-1] -
                             flux_down_diffuse_pred[:,1:] + 
                             flux_up_diffuse_pred[:,1:] -
                            flux_up_diffuse_pred[:,:-1])                        
                         
                         
    hr_diffuse_loss = loss_heating_rate_full(flux_absorbed_diffuse_true, 
                                             flux_absorbed_diffuse_pred,toa_weighting_profile, delta_pressure)

    return hr_diffuse_loss

def loss_heating_rate_diffuse_wrapper(data, y_pred):
    _, _, toa, delta_pressure, y_true, _ = data
    loss = loss_heating_rate_diffuse(y_true, y_pred, toa, 
               delta_pressure)
    return loss

def loss_flux_diffuse(y_true, y_pred, toa_weighting_profile):
    # Handles flux using TOA weight rather than weight profile
    
    (_, flux_down_diffuse_pred, 
     flux_up_diffuse_pred, _) = y_pred
    
    (flux_down_direct_true, flux_down_true, 
     flux_up_diffuse_true) = (y_true[:,:,0], y_true[:,:,1], y_true[:,:,2])
    flux_down_diffuse_true = flux_down_true - flux_down_direct_true
    
    flux_diffuse_pred = torch.concat((flux_down_diffuse_pred,flux_up_diffuse_pred),dim=1)
    flux_diffuse_true = torch.concat((flux_down_diffuse_true,flux_up_diffuse_true),dim=1)

    flux_diffuse_loss = loss_weighted(flux_diffuse_true, flux_diffuse_pred, toa_weighting_profile)

    return flux_diffuse_loss

def loss_flux_diffuse_wrapper(data, y_pred):
    _, _, toa, _, y_true, _ = data
    loss = loss_flux_diffuse(y_true, y_pred, toa)
    return loss

def loss_henry_2(y_true, y_pred, toa_weighting_profile, 
               delta_pressure):
    # Handles flux using TOA weight rather than weight profile
    
    (flux_down_direct_pred, flux_down_diffuse_pred, 
     flux_up_diffuse_pred, flux_absorbed_pred) = y_pred
    
    (flux_down_direct_true, flux_down_true, 
     flux_up_diffuse_true) = (y_true[:,:,0], y_true[:,:,1], y_true[:,:,2])
    flux_down_diffuse_true = flux_down_true - flux_down_direct_true
    
    flux_diffuse_pred = torch.concat((flux_down_diffuse_pred,flux_up_diffuse_pred),dim=1)
    flux_diffuse_true = torch.concat((flux_down_diffuse_true,flux_up_diffuse_true),dim=1)

    flux_diffuse_loss = loss_weighted(flux_diffuse_true, flux_diffuse_pred, toa_weighting_profile)
    flux_direct_loss = loss_weighted(flux_down_direct_true, flux_down_direct_pred, toa_weighting_profile)

    flux_absorbed_diffuse_true = (flux_down_diffuse_true[:,:-1] -
                             flux_down_diffuse_true[:,1:] + 
                             flux_up_diffuse_true[:,1:] -
                             flux_up_diffuse_true[:,:-1])


    flux_absorbed_diffuse_pred = (flux_down_diffuse_pred[:,:-1] -
                             flux_down_diffuse_pred[:,1:] + 
                             flux_up_diffuse_pred[:,1:] -
                            flux_up_diffuse_pred[:,:-1])                        
                         
    hr_diffuse_loss = loss_heating_rate_full(flux_absorbed_diffuse_true, 
                                             flux_absorbed_diffuse_pred,toa_weighting_profile, delta_pressure)
    
    hr_direct_loss = loss_heating_rate_direct(
        flux_down_direct_true, flux_down_direct_pred, 
        toa_weighting_profile, delta_pressure)

    if True:
        # v18 0-145
        # v19, v22 0-195, v27, v28 0-275, v29 0-310
        #hr_weight   = 0.3
        #direct_weight = 3.0
        # v19 195-, v28 275-325, v29 310 - 395
        #hr_weight   = 0.37
        #direct_weight = 1.5
        # v19 400-, v28 325-, v29 395-
        hr_weight   = 0.5
        direct_weight = 1.0
        return (1.0 / (1.0 + direct_weight)) * (hr_weight * 
                         (hr_diffuse_loss + direct_weight * hr_direct_loss) + (1.0 - hr_weight) * (flux_diffuse_loss + direct_weight * flux_direct_loss))


def loss_henry_full_wrapper(data, y_pred):
    _, _, toa, delta_pressure, y_true, _ = data
    loss = loss_henry_2(y_true, y_pred, toa, 
               delta_pressure)
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
    train_input_dir = "/data-T1/hws/CAMS/processed_data/training/2008/"
    cross_input_dir = "/data-T1/hws/CAMS/processed_data/cross_validation/2008/"
    months = [str(m).zfill(2) for m in range(1,13)]
    train_input_files = [f'{train_input_dir}Flux_sw-2008-{month}.nc' for month in months]
    cross_input_files = [f'{cross_input_dir}Flux_sw-2008-{month}.nc' for month in months]

    #batch_size = 2048
    #batch_size = 1536 
    batch_size = 1024
    n_channel = 42
    n_constituent = 8
    #filename_full_model = datadir + f"/Torch.Dataloader.v2_1." # scattering_v3
    # log_1.txt, log_2.txt
    #filename_full_model = datadir + f"/Torch.Dataloader.v2_2." # scattering_v3
    # Uses flat correction for flux weighting, log_v2.txt

    #filename_full_model = datadir + f"/Torch.Dataloader.v2_3." # scattering_v3
    # Uses flat correction for flux weighting * 3.0
    # log_v3.txt - initialization
    # log_v3_1 

    #filename_full_model = datadir + f"/Torch.Dataloader.v2_4." # scattering_v3
    # Uses loss_henry_2 with 0.667 weight on heating and 0.33 on flux
    # log_v4-i.txt - initialization
    # log_v4 
    # log_v4_results

    #filename_full_model = datadir + f"/Torch.Dataloader.v2_5." # scattering_v3
    # Uses loss_henry_2 with 0.3 weight on heating and 0.7 on flux
    # 2X weight on direct
    # log_v5-i.txt - initialization
    # log_v5 
    # log_v5_results

    #filename_full_model = datadir + f"/Torch.Dataloader.v3_6." # scattering_v3
    # Uses loss_henry_2 with 0.3 weight on heating and 0.7 on flux
    # 2X weight on direct
    # total loss * 0.3333
    # 42 channels
    # log_v6-i.txt - initialization
    # log_v6 
    # log_v6_results

    #filename_full_model = datadir + f"/Torch.Dataloader.v3_7." # scattering_v3
    # Same as v3_6 except us Scattering_v2_tau
    # log_v7-i.txt - initialization
    # log_v7 
    # log_v7_results

    #filename_full_model = datadir + f"/Torch.Dataloader.v3_8." # scattering_v3
    # Same as v3_7 except uses Scattering_v2_tau_efficient
    # log_v8-i.txt - initialization
    # log_v8 
    # log_v8_results

    #filename_full_model = datadir + f"/Torch.Dataloader.v3_9." # scattering_v3
    # Same as v3_8 except uses Extinction_Efficient
    # log_v9-i.txt - initialization
    # log_v9 
    # log_v9_results

    #filename_full_model = datadir + f"/Torch.Dataloader.v3_10." # scattering_v3
    # Same as v3_8 except uses Scattering_v2_tau_efficient_2
    # log_v10-i.txt - initialization
    # log_v10 
    # log_v10_results
    # SLOW!

    #filename_full_model = datadir + f"/Torch.Dataloader.v3_11." # scattering_v3
    # Same as v3_7 except uses Scattering_v2_tau_efficient with more
    # aggressive settings
    # log_v11-i.txt - initialization
    # log_v11 
    # log_v11_results

    #filename_full_model = datadir + f"/Torch.Dataloader.v3_12." # scattering_v3
    # Same as v3_11 except uses BD instead MLP
    # log_v12-i.txt - initialization
    # log_v12 
    # log_v12_results    
    # 
    #filename_full_model = datadir + f"/Torch.Dataloader.v3_13." # scattering_v3
    # Same as v3_12 except uses different BD configuration
    # log_v13-i.txt - initialization
    # log_v13 
    # log_v13_results

    #filename_full_model = datadir + f"/Torch.Dataloader.v3_14." # scattering_v2_efficient
    # Same as v3_13 except uses bias=True in BD and gives 4.0 weight
    # to direct terms in henry_loss_2
    # log_v14-i.txt - initialization
    # log_v14 
    # log_v14_results

    #filename_full_model = datadir + f"/Torch.Dataloader.v3_15." # scattering_v2_efficient
    # Same as v3_14 except uses standard loss and lr=0.001
    # to direct terms in henry_loss_2
    # log_v15-i.txt - initialization
    # log_v15 
    # log_v15_results

    #filename_full_model = datadir + f"/Torch.Dataloader.v3_16." # scattering_v2_efficient
    # Same as v3_15 except uses uses loss with focussed on direct x 1000
    # to direct terms in henry_loss_2
    # log_v16-i.txt - initialization
    # log_v16 
    # log_v16_results

    #filename_full_model = datadir + f"/Torch.Dataloader.v3_17." # scattering_v2_efficient
    # Same as v3_16 except uses uses only direct loss upto iteration 350
    # to direct terms in henry_loss_2
    # log_v17-i.txt - initialization
    # log_v17 
    # log_v17_results

    #filename_full_model = datadir + f"/Torch.Dataloader.v5_18." # scattering_v2_efficient
    # Similiar to v15 except separates out direct and diffuse losses
    # to direct terms in henry_loss_2
    # log_v18-i.txt - initialization
    # log_v18 
    # log_v18_results

    #filename_full_model = datadir + f"/Torch.Dataloader.v5_19." # scattering_v2_efficient
    # Similiar to v18 except uses differences for computing flux_absorbed_diffuse_pred

    #filename_full_model = datadir + f"/Torch.Dataloader.v5_22." # scattering_v2_efficient
    # Similiar to v19, except changes eps to 0.002

    #filename_full_model = datadir + f"/Torch.Dataloader.v5_23." # scattering_v2_efficient
    # Similiar to v19, except changes eps to 0.01
    
    #filename_full_model = datadir + f"/Torch.Dataloader.v5_24." # scattering_v2_efficient
    # Similiar to v19, except uses total heat rate cost function

    #filename_full_model = datadir + f"/Torch.Dataloader.v5_25." # scattering_v2_efficient
    # Similiar to v19, except uses total heat rate cost function and total flux cost function

    #filename_full_model = datadir + f"/Torch.Dataloader.v5_26." # scattering_v2_efficient
    # Restarts v25 at iteration 105 using more aggressive dropout

    #filename_full_model = datadir + f"/Torch.Dataloader.v5_27." # scattering_v2_efficient
    # Same as v19 (including loss function), but uses revised to Extinction to impose dropout 
    # on "effective masses"

    #filename_full_model = datadir + f"/Torch.Dataloader.v5_28." # scattering_v2_efficient
    # Same as v19 (including loss function), but uses exp(-tau) as input to scattering

    filename_full_model = datadir + f"/Torch.Dataloader.v5_29." # scattering_v2_efficient
    # Same as v19 (including loss function), but does not divide tau by mu for input to direct scattering. 
    # Instead adds 1/mu as input to scattering

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
            #initial_model_n = 0  #4
            #initial_model_n = 5   #5
            #initial_model_n = 1   #6
            #initial_model_n = 3   #7
            #initial_model_n = 1   #8
            #initial_model_n = 8   #9
            #initial_model_n = 2   #11
            #initial_model_n = 2   #12
            #initial_model_n = 3   #13
            #initial_model_n = 0   #14
            #initial_model_n = 2   #15
            #initial_model_n = 2   #17
            #initial_model_n = 0   #18
            #initial_model_n = 0   #19
            #initial_model_n = 1   #22
            #initial_model_n = 3   #24
            #initial_model_n = 1   #25
            #initial_model_n = 3   #28
            initial_model_n = 4   #29
            t_start = 1
            filename_full_model_input = f'{filename_full_model}i' + str(initial_model_n).zfill(2)
        else:
            t_start = 560 #395
            filename_full_model_input = filename_full_model + str(t_start).zfill(3)


    for ee in range(number_of_tries):

        #print(f"Model = {str(ee).zfill(3)}")


        #filename_full_model = filename_full_model_input  #

        t_warmup = 1  # for profiling
        t = t_start
        dropout_p = 0.00

        dropout_schedule = (0.0, 0.07, 0.1, 0.15, 0.2, 0.15, 0.1, 0.07, 0.0, 0.0) 

        # Used for v11
        #dropout_schedule = (0.0, 0.07, 0.0, 0.0) 

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
        dropout_epochs =   (-1, 40, 60, 70,  80, 90,  105, 120, 135, epochs + 1)

        #V26
        #dropout_epochs =   (-1, 40, 60, 70,  105, 140,  150, 160, 170, epochs + 1)

        # Used for v11
        #dropout_epochs =   (-1, 65, 85, epochs + 1)

        # v28
        #dropout_epochs =   (-1, 80, 100, 130,  150, 165,  175, 185, 195, epochs + 1) 

        dropout_index = next(i for i, x in enumerate(dropout_epochs) if t <= x) - 1
        dropout_p = dropout_schedule[dropout_index]
        last_dropout_index = dropout_index

        model = FullNet(n_channel,n_constituent,dropout_p,device).to(device=device)

        n_parameters = count_parameters(model)

        print(f"Number of parameters = {n_parameters}")

        if is_initial_condition:
                n_parameters = count_parameters(model)

                t1 = count_parameters(model.mu_diffuse_net) + count_parameters(model.spectral_net)
                print(f"Number of diffuse, spectral = {t1}")

                t2 = count_parameters(model.extinction_net)
                print(f"Number of extinction = {t2}")

                t3 = count_parameters(model.scattering_net)
                print(f"Number of scattering = {t3}")

                if False:
                    t4 = count_parameters(model.scattering_net.direct_scattering)
                    print(f"Number of scattering - direct scattering = {t4}")

                    t5 = count_parameters(model.scattering_net.direct_selection)
                    print(f"Number of scattering - direct selection = {t5}")

        # v7 @ 295 changed lr=0.0025
        # v7 @ 320 changed lr=0.0075
        # v11 @ 40 changed lr=0.0075 to 0.002
        # v11 @ 65 changed lr from 0.002 to 0.001
        # v11 @ 140 changed lr from 0.001 to 0.0003
        # v11 @ 160 changed lr from 0.0003 to 0.0001
        # v12 @ 1 lr=0.001
        # v12 @ 45 lr= 0.004
        # v12 @ 145 lr= 0.008
        # v12 @ 200 lr= 0.016
        # v13 @ 1 lr=0.002
        # v14 @ 65 lr = 0.001
        # v19 0-250 lr=0.001, 250+ lr=0.00038
        # v22 -v25 lr = 0.001
        # v26 lr = 0.002 for 105 - 115, lr = 0.001 otherwise
        # v29 lr = 0.001 for 0-320, lr=0.00038 320-410, lr=0.00013 410+
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00013)

        train_dataset = RT_data_hws_2.RTDataSet(train_input_files,n_channel)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, 
                                                    shuffle=False,
                                                            num_workers=1)
        
        validation_dataset = RT_data_hws_2.RTDataSet(cross_input_files,n_channel)

        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size, 
                                                    shuffle=False,
                                                            num_workers=1)
        
        #start = torch.cuda.Event(enable_timing=True)
        #end = torch.cuda.Event(enable_timing=True)

        #loss_functions = (loss_henry_full_wrapper, loss_flux_full_wrapper_2, loss_heating_rate_full_wrapper,
                        #loss_heating_rate_direct_full_wrapper, loss_flux_full_wrapper)
                     
        #loss_names = ("Loss", "Flux Loss (weight profile)", "Heating Rate Loss", 
                    #"Direct Heating Rate Loss", "Flux Loss (TOA weighting)")
                        

        loss_functions = (loss_henry_full_wrapper, loss_flux_full_wrapper, loss_flux_direct_wrapper_2, 
                          loss_flux_diffuse_wrapper, loss_heating_rate_full_wrapper,
                        loss_heating_rate_direct_full_wrapper, 
                        loss_heating_rate_diffuse_wrapper,
                        )
        loss_names = ("Loss", "Flux Loss", "Flux Loss Direct", "Flux Loss Diffuse", "Heating Rate Loss", 
                    "Heating Rate Loss Direct", "Heating Rate Loss Diffuse"
                        )

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
                           #loss_heat_flux_full_wrapper,
                           #loss_heating_rate_full_wrapper, #
                           loss_henry_full_wrapper, 
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
    batch_size = 1048 
    n_channel = 42 #30
    n_constituent = 8
    is_use_internals = True #False #True

    if is_use_internals:
        model = FullNetInternals(n_channel,n_constituent,dropout_p=0,device=device)
    else:
        model = FullNet(n_channel,n_constituent,dropout_p=0,device=device)

    print(model)

    n_parameters = count_parameters(model)

    t1 = count_parameters(model.mu_diffuse_net) + count_parameters(model.spectral_net)
    print(f"Number of diffuse, spectral = {t1}")

    t2 = count_parameters(model.extinction_net)
    print(f"Number of extinction = {t2}")

    t3 = count_parameters(model.scattering_net)
    print(f"Number of scattering = {t3}")


    model = model.to(device=device)


    #filename_full_model = datadir + "/Torch.Dataloader.1/Torch.Dataloader.1." # Scattering_v3
    #filename_full_model = datadir + "/Torch.Dataloader.e8/Torch.Dataloader.e8." # Scattering_v3
    #filename_full_model = datadir + f"/Torch.Dataloader.4/Torch.Dataloader.4." # corresponds to Scattering_v1_tau

    #filename_full_model = datadir + f"/Torch.Dataloader.v2_4." # corresponds to Scattering_v3, two inputs to mass_extinction(t,p)

    #filename_full_model = datadir + f"/Torch.Dataloader.v2_5." # corresponds to Scattering_v3, two inputs to mass_extinction(t,p)

    #filename_full_model = datadir + f"/Torch.Dataloader.v3_6." # scattering_v3
    # Uses loss_henry_2 with 0.3 weight on heating and 0.7 on flux
    # 2X weight on direct
    # total loss * 0.3333
    # 42 channels
    # log_v6-i.txt - initialization
    # log_v6 
    # log_v6_results

    #filename_full_model = datadir + f"/Torch.Dataloader.v3_7." # scattering_v3
    # Same as v3_6 except us Scattering_v2_tau
    # log_v7-i.txt - initialization
    # log_v7 
    # log_v7_results

    #filename_full_model = datadir + f"/Torch.Dataloader.v3_11." # scattering_v3
    # Same as v3_7 except uses Scattering_v2_tau_efficient with more
    # aggressive settings
    # log_v11-i.txt - initialization
    # log_v11 
    # log_v11_results

    #filename_full_model = datadir + f"/Torch.Dataloader.v3_15." # scattering_v2_efficient
    # Same as v3_14 except uses standard loss and lr=0.001
    # to direct terms in henry_loss_2
    # log_v15-i.txt - initialization
    # log_v15 
    # log_v15_results


    #filename_full_model = datadir + f"/Torch.Dataloader.v3_17." # scattering_v2_efficient
    # Same as v3_14 except uses standard loss and lr=0.001
    # to direct terms in henry_loss_2
    # log_v17-i.txt - initialization
    # log_v17 
    # log_v17_results


    #filename_full_model = datadir + f"/Torch.Dataloader.v5_19." # scattering_v2_efficient
    # Similiar to v18 except uses differences for computing flux_absorbed_diffuse_pred


    #filename_full_model = datadir + f"/Torch.Dataloader.v5_24." # scattering_v2_efficient
    # Similiar to v19, except uses total heat rate cost function

    #version_name = 'v5_25'
    #filename_full_model = datadir + f"/Torch.Dataloader.{version_name}." # scattering_v2_efficient
    # Similiar to v19, except uses total heat rate cost function and total flux cost function

    #version_name = 'v5_28'
    #filename_full_model = datadir + f"/Torch.Dataloader.v5_28." # scattering_v2_efficient
    # Same as v19 (including loss function), but uses exp(-tau) as input to scattering

    version_name = 'v5_29'
    filename_full_model = datadir + f"/Torch.Dataloader.v5_29." 

    years = ("2009", "2015", "2020")
    #years = ("2020", )

    for year in years:
        test_input_dir = f"/data-T1/hws/CAMS/processed_data/testing/{year}/"
        months = [str(m).zfill(2) for m in range(1,13)]
        test_input_files = [f'{test_input_dir}Flux_sw-{year}-{month}.nc' for month in months]
        #test_input_files = ["/data-T1/hws/tmp/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.2.nc"]


        test_dataset = RT_data_hws_2.RTDataSet(test_input_files,n_channel)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, 
                                                    shuffle=False,
                                                            num_workers=1)


        #loss_functions = (loss_henry_full_wrapper, loss_heating_rate_full_wrapper,
        #                loss_heating_rate_direct_full_wrapper, loss_flux_full_wrapper)
        #loss_names = ("Loss", "Heating Rate Loss", "Direct Heating Rate Loss", "Flux Loss")



        loss_functions = (loss_henry_full_wrapper, loss_flux_full_wrapper, loss_flux_direct_wrapper_2, 
                          loss_flux_diffuse_wrapper, loss_heating_rate_full_wrapper,
                        loss_heating_rate_direct_full_wrapper, 
                        loss_heating_rate_diffuse_wrapper,
                        )
        loss_names = ("Loss", "Flux Loss", "Flux Loss Direct", "Flux Loss Diffuse", "Heating Rate Loss", 
                    "Heating Rate Loss Direct", "Heating Rate Loss Diffuse"
                        )

        print(f"Testing error, Year = {year}")
        for t in range(650,655,5):

            checkpoint = torch.load(filename_full_model + str(t), map_location=torch.device(device))
            print(f"Loaded Model: epoch = {t}")
            model.load_state_dict(checkpoint['model_state_dict'])

            print(f"Total number of parameters = {n_parameters}", flush=True)
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

    def _adding_doubling (self, t_direct, t_diffuse, 
                        e_split_direct, e_split_diffuse, 
                        r_surface_direct, r_surface_diffuse, 
                        a_surface_direct, a_surface_diffuse):
        """
       
        """
        # Split out extinguished component
        e_direct = 1.0 - t_direct
        e_diffuse = 1.0 - t_diffuse

        # Split extinguished into transmitted, reflected, and absorbed
        e_t_direct, e_r_direct, e_a_direct = (e_split_direct[:,:,0], 
                                              e_split_direct[:,:,1],
                                              e_split_direct[:,:,2])
        e_t_diffuse, e_r_diffuse, e_a_diffuse = (e_split_diffuse[:,:,0], 
                                                 e_split_diffuse[:,:,1],
                                                 e_split_diffuse[:,:,2])

        eps = 1.0e-06
        d = 1.0/(1.0 - e_diffuse*e_r_diffuse*r_surface_diffuse + eps)

        # Adding-Doubling for direct radiation
        t_multi_direct = (t_direct* r_surface_direct * e_diffuse * e_r_diffuse*d 
                        + e_direct * e_t_direct * d)
        
        a_surface_multi_direct = (t_direct * a_surface_direct 
                                + t_multi_direct * a_surface_diffuse)

        r_surface_multi_direct = (t_direct * r_surface_direct * d 
                                + e_direct * e_t_direct * r_surface_diffuse * d)

        a_layer_multi_direct = (e_direct * e_a_direct 
                            + r_surface_multi_direct * e_diffuse * e_a_diffuse)

        r_layer_multi_direct = (e_direct * e_r_direct 
                        + r_surface_multi_direct 
                        * (t_diffuse + e_diffuse * e_t_diffuse))

        # Adding-Doubling for diffuse radiation
        t_multi_diffuse = (
            t_diffuse * r_surface_diffuse * e_diffuse * e_r_diffuse * d 
            + e_diffuse * e_t_diffuse * d) 
        
        a_surface_multi_diffuse = (t_diffuse * a_surface_diffuse 
                                + t_multi_diffuse * a_surface_diffuse)

        r_surface_multi_diffuse = (t_diffuse * r_surface_diffuse * d 
                             + e_diffuse * e_t_diffuse * r_surface_diffuse*d)
        
        a_layer_multi_diffuse = (e_diffuse * e_a_diffuse 
                        + r_surface_multi_diffuse * e_diffuse * e_a_diffuse)

        r_layer_multi_diffuse = (e_diffuse * e_r_diffuse 
                        + r_surface_multi_diffuse 
                        * (t_diffuse + e_diffuse * e_t_diffuse))

        return (t_multi_direct, t_multi_diffuse, 
                r_layer_multi_direct, r_layer_multi_diffuse, 
                r_surface_multi_direct, r_surface_multi_diffuse, 
                a_layer_multi_direct, a_layer_multi_diffuse, 
                a_surface_multi_direct, a_surface_multi_diffuse)
