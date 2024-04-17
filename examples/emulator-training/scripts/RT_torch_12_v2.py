# Same as RT_torch_12 except mass extinction depends only on (T,P) 
# instead of (T,P,ln(P))
# Added variants for cost function, cost_henry_2 with various weightings
# of flux and heating rate and greater weight for direct flux, direct heat

import numpy as np
import time
from typing import List
import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F

from RT_data_hws import load_data_direct_pytorch, load_data_full_pytorch_2, absorbed_flux_to_heating_rate
import RT_data_hws_2

eps_1 = 0.0000001

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
                 lower=-0.1, upper=0.1):
        super(MLP, self).__init__()
        self.n_hidden = n_hidden
        self.n_outputs = n_output
        n_last = n_input
        self.hidden = nn.ModuleList()

        for n in n_hidden:
            mod = nn.Linear(n_last, n, bias=True,device=device)
            torch.nn.init.uniform_(mod.weight, a=lower, b=upper)
            # Bias initialized to ~1.0
            # Because of ReLU activation, don't want any connections to
            # be prematurely pruned away by becoming negative.
            # Therefore start with a significant positive bias
            torch.nn.init.uniform_(mod.bias, a=0.9, b=1.1) #a=-0.1, b=0.1)
            self.hidden.append(mod)
            n_last = n
        self.dropout_p = dropout_p
        self.output = nn.Linear(n_last, n_output, bias=True, device=device)
        torch.nn.init.uniform_(self.output.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.output.bias, a=-0.1, b=0.1)

    def reset_dropout(self,dropout_p):
        self.dropout_p = dropout_p

    #@torch.compile
    def forward(self, x):
        for hidden in self.hidden:
            x = hidden(x)
            x = F.dropout(x,p=self.dropout_p,training=self.training)
            x = F.relu(x)
        return self.output(x)

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

        # Computes a scalar extinction coeffient for each constituent 
        # for each channel
        self.net_lw  = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_iw  = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_h2o = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_o3  = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_co2 = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_u   = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_n2o = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_ch4 = nn.Linear(1,self.n_channel,bias=False,device=device)

        lower = -0.9 # exp(-0.9) = .406
        upper = 0.5  # exp(0.5) = 1.64
        torch.nn.init.uniform_(self.net_lw.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_iw.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_h2o.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_o3.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_co2.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_u.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_n2o.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_ch4.weight, a=lower, b=upper)

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
        self.net_ke_u   = MLP(n_input=2,n_hidden=(6,4,4),n_output=1,
                              dropout_p=dropout_p,device=device)
        self.net_ke_n2o = MLP(n_input=2,n_hidden=(6,4,4),n_output=1,
                              dropout_p=dropout_p,device=device)
        self.net_ke_ch4 = MLP(n_input=2,n_hidden=(6,4,4),n_output=1,
                              dropout_p=dropout_p,device=device)

        # Filters select which channels each constituent contributes to
        # Follows similiar assignment of bands as
        # Table A2 in Pincus, R., Mlawer, E. J., &
        # Delamere, J. S. (2019). Balancing accuracy, efficiency, and 
        # flexibility in radiation calculations for dynamical models. Journal 
        # of Advances in Modeling Earth Systems, 11,3074–3089. 
        # https://doi.org/10.1029/2019MS001621

        self.filter_h2o = torch.tensor([1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
                                       1,1,1,1,1, 1,1,0,0,0, 0,0,0,1,1,],
                                       dtype=torch.float32,device=device)

        self.filter_o3 = torch.tensor([1,1,0,0,0, 0,0,0,0,0, 0,0,0,0,0,
                                       0,1,1,1,1, 1,1,0,0,1, 1,1,1,1,1,],
                                       dtype=torch.float32,device=device)

        self.filter_co2 = torch.tensor([1,1,0,0,1, 1,0,0,1,1, 0,0,1,1,0,
                                       0,0,0,0,0, 0,0,0,0,0, 0,0,0,1,1,],
                                       dtype=torch.float32,device=device)
        
        self.filter_u  = torch.tensor([1,1,0,0,0, 0,0,0,0,0, 0,0,0,0,1,
                                       1,1,1,1,1, 1,1,0,0,0, 0,1,1,1,1,],
                                       dtype=torch.float32,device=device)

        self.filter_n2o = torch.tensor([1,1,0,0,1, 0,0,0,0,0, 0,0,0,0,0,
                                       0,0,0,0,0, 0,0,0,0,0, 0,0,0,1,1,],
                                       dtype=torch.float32,device=device)
        
        self.filter_ch4 = torch.tensor([1,1,1,1,0, 0,1,1,0,0, 1,1,0,0,0,
                                       0,0,0,0,0, 0,0,0,0,0, 0,0,0,1,1,],
                                       dtype=torch.float32,device=device)
    def reset_dropout(self,dropout_p):
        self.net_ke_h2o.reset_dropout(dropout_p)
        self.net_ke_o3.reset_dropout(dropout_p)
        self.net_ke_co2.reset_dropout(dropout_p)
        self.net_ke_u.reset_dropout(dropout_p)
        self.net_ke_n2o.reset_dropout(dropout_p)
        self.net_ke_ch4.reset_dropout(dropout_p)

    def forward(self, x):
        temperature_pressure_log_pressure, constituents = x

        c = constituents
        t_p = temperature_pressure_log_pressure[:,:2]  #Removing ln-P

        a = torch.exp
        b = torch.sigmoid

        one = torch.ones((c.shape[0],1),dtype=torch.float32,device=self.device)
        tau_lw  = a(self.net_lw (one)) * (c[:,0:1])
        tau_iw  = a(self.net_iw (one)) * (c[:,1:2])
        tau_h2o = a(self.net_h2o(one)) * (c[:,2:3]) * (self.filter_h2o * 
                                                       b(self.net_ke_h2o(t_p)))
        tau_o3  = a(self.net_o3 (one)) * (c[:,3:4]) * (self.filter_o3  * 
                                                       b(self.net_ke_o3 (t_p)))
        tau_co2 = a(self.net_co2(one)) * (c[:,4:5]) * (self.filter_co2 * 
                                                       b(self.net_ke_co2(t_p)))
        tau_u   = a(self.net_u  (one)) * (c[:,5:6]) * (self.filter_u * 
                                                       b(self.net_ke_u  (t_p)))
        tau_n2o = a(self.net_n2o(one)) * (c[:,6:7]) * (self.filter_n2o * 
                                                       b(self.net_ke_n2o(t_p)))
        tau_ch4 = a(self.net_ch4(one)) * (c[:,7:8]) * (self.filter_ch4 * 
                                                       b(self.net_ke_ch4(t_p)))

        tau_lw  = torch.unsqueeze(tau_lw,2)
        tau_iw  = torch.unsqueeze(tau_iw,2)
        tau_h2o = torch.unsqueeze(tau_h2o,2)
        tau_o3  = torch.unsqueeze(tau_o3,2)
        tau_co2 = torch.unsqueeze(tau_co2,2)
        tau_u   = torch.unsqueeze(tau_u,2)
        tau_n2o = torch.unsqueeze(tau_n2o,2)
        tau_ch4 = torch.unsqueeze(tau_ch4,2)

        tau = torch.cat([tau_lw, tau_iw, tau_h2o, tau_o3, tau_co2, tau_u, 
                         tau_n2o, tau_ch4],dim=2)

        return tau


    
class DirectTransmission(nn.Module):
    """ Only Computes Direct Transmission Coefficient """
    def __init__(self):
        super(DirectTransmission, self).__init__()

    def forward(self, x):
        mu_direct, tau = x
        
        tau_total = torch.sum(tau,dim=2,keepdim=False)
        t_direct = torch.exp(-tau_total / (mu_direct + eps_1))
        return t_direct


class Scattering_v1(nn.Module):
    """ 
    For a given atmospheric layer, learns the split of extinguished
    radiation into transmitted, reflected, and absorbed components
    using MLP modules.
     
    Learns these separately for direct and diffuse 
    input and then separately for each channel.

    Computes the direct transmission coefficients
    by taking the exponential of the negative optical depth
    scaled by the cosine of the zenith angle (Beer's Law)

    Note: The suffixes "_direct" and "_diffuse" specify the type of 
    input radiation.

    Note: Consider re-doing with tau as input to the scattering net similar
    to v3

    Inputs:

        mu_direct, mu_diffuse: cosine of zenith angle

        tau: optical depth of each constituent for each channel 

        constituents: mass of each constituent in layer

    Outputs:

        t_direct, t_diffuse: direct transmission coefficients of 
            the layer. Note: t_diffuse accounts for diffuse radiation that
            is directly transmitted.

        e_split_direct, e_split_diffuse: the split of extinguised  
            radiation into transmitted, reflected, and absorbed components. 
            The last dimension in each tensor has 3 elements corresponding 
            to these components. For each data point these components sum 
            to 1.0.
            
    """

    def __init__(self, n_channel, n_constituent, dropout_p, device):

        super(Scattering_v1, self).__init__()
        self.n_channel = n_channel

        n_hidden = [5, 4, 4]
        # For direct input
        # Has additional input for zenith angle ('mu_direct')
        self.net_direct = nn.ModuleList(
            [MLP(n_input=n_constituent + 1,
                 n_hidden=n_hidden,
                 n_output=3,
                 dropout_p=dropout_p,
                 device=device,
                 lower=-1.0,upper=1.0) 
             for _ in range(self.n_channel)])
        # For diffuse input
        self.net_diffuse = nn.ModuleList(
            [MLP(n_input=n_constituent, 
                 n_hidden=n_hidden, 
                 n_output=3,
                dropout_p=dropout_p,
                 device=device, 
                 lower=-1.0,upper=1.0) 
             for _ in range(self.n_channel)])

    def reset_dropout(self,dropout_p):
        for net in self.net_direct:
            net.reset_dropout(dropout_p)
        for net in self.net_diffuse:
            net.reset_dropout(dropout_p)

    def forward(self, x):
        tau, mu_direct, mu_diffuse, constituents = x

        tau_total = torch.sum(tau, dim=2, keepdims=False)

        t_direct = torch.exp(-tau_total / (mu_direct + eps_1))
        t_diffuse = torch.exp(-tau_total / (mu_diffuse + eps_1))

        constituents_direct = constituents / (mu_direct + eps_1)
        constituents_diffuse = constituents 
        constituents_direct = torch.concat((constituents_direct, mu_direct),
                                           dim=1)

        e_split_direct = [F.softmax(net(constituents_direct),dim=-1) for net 
                          in self.net_direct]
        e_split_diffuse = [F.softmax(net(constituents_diffuse),dim=-1) for net 
                           in self.net_diffuse]

        e_split_direct = torch.stack(e_split_direct, dim=1)
        e_split_diffuse = torch.stack(e_split_diffuse, dim=1)

        layers = [t_direct, t_diffuse, 
                  e_split_direct, e_split_diffuse]

        return layers
    

class Scattering_v1_tau(nn.Module):
    """ 
    Same as V1 except instead of using the constituents, combines 
    the tau's as follows, separating those that scatter from 
    those that don't.
    Inputs: tau's of lwp, iwp, sum(o3,co2,n2o,ch4), h2o, u
    Goal is to separate scatterers from absorbers
    """

    def __init__(self, n_channel, n_constituent, dropout_p, device):

        super(Scattering_v1_tau, self).__init__()
        self.n_channel = n_channel

        n_input = n_constituent #5

        n_hidden = [5, 4, 4]
        # For direct input
        # Has additional input for zenith angle ('mu_direct')
        self.net_direct = nn.ModuleList(
            [MLP(n_input=n_input + 1,
                 n_hidden=n_hidden,
                 n_output=3,
                 dropout_p=dropout_p,
                 device=device,
                 lower=-1.0,upper=1.0) 
             for _ in range(self.n_channel)])
        # For diffuse input
        self.net_diffuse = nn.ModuleList(
            [MLP(n_input=n_input, 
                 n_hidden=n_hidden, 
                 n_output=3,
                dropout_p=dropout_p,
                 device=device, 
                 lower=-1.0,upper=1.0) 
             for _ in range(self.n_channel)])

    def reset_dropout(self,dropout_p):
        for net in self.net_direct:
            net.reset_dropout(dropout_p)
        for net in self.net_diffuse:
            net.reset_dropout(dropout_p)

    def forward(self, x):
        tau, mu_direct, mu_diffuse, _ = x

        #print(f"tau.shape = {tau.shape}")
        tau_total = torch.sum(tau, dim=2, keepdims=False)

        t_direct = torch.exp(-tau_total / (mu_direct + eps_1))
        t_diffuse = torch.exp(-tau_total / (mu_diffuse + eps_1))

        mu_direct = torch.unsqueeze(mu_direct,dim=1)
        tau_direct = tau / (mu_direct + eps_1)

        #direct_1 = tau_direct[:,:,3] + tau_direct[:,:,4]+ tau_direct[:,:,6] + tau_direct[:,:,7]

        #direct_1 = torch.unsqueeze(direct_1,dim=2)

        mu_direct = mu_direct.repeat(1,self.n_channel,1)
        #direct = torch.concat((tau_direct[:,:,0:2], tau_direct[:,:,5:6],tau_direct[:,:,2:3], direct_1, mu_direct), dim=2)

        #print(f"tau_direct.shape = {tau_direct.shape}")
        direct = torch.concat((tau_direct, mu_direct),
                                           dim=2)

        #diffuse_1 = tau[:,:,3] + tau[:,:,4]+ tau[:,:,6] + tau[:,:,7]

        #diffuse_1 = torch.unsqueeze(diffuse_1,dim=2)

        #diffuse = torch.concat((tau[:,:,0:2], tau[:,:,5:6], tau[:,:,2:3], diffuse_1), dim=2)
        
        e_split_direct = [F.softmax(net(direct[:,i,:]),dim=-1) for i, net 
                          in enumerate(self.net_direct)]

        #print(f"len of e_split_direct = {len(e_split_direct)}")
        #print(f"shape of e_split_direct[0] = {e_split_direct[0].shape}", flush=True)

        e_split_diffuse = [F.softmax(net(tau[:,i,:]),dim=-1) for i, net 
                           in enumerate(self.net_diffuse)]

        e_split_direct = torch.stack(e_split_direct, dim=1)
        e_split_diffuse = torch.stack(e_split_diffuse, dim=1)

        layers = [t_direct, t_diffuse, 
                  e_split_direct, e_split_diffuse]

        return layers

class Scattering_v2(nn.Module):
    """ Computes full set of layer properties for
    direct and diffuse transmission, reflection,
    and absorption
    Uses a single net for scattering for all channels
    Uses constituent (gas concentration) values as input

    *** No longer used ****
    """
    def __init__(self, n_channel, n_constituent, n_coarse_code, dropout_p, device):
        super(Scattering_v2, self).__init__()
        self.n_channel = n_channel
        self.n_coarse_code = n_coarse_code
        n_hidden = [10, 6, 6, 6]
        """ Computes split of extinguished radiation into absorbed, diffuse transmitted, and
        diffuse reflected """
        self.net_direct = MLP(n_input=n_constituent + n_coarse_code + 1, 
                 n_hidden=n_hidden, 
                 n_output=3,
                dropout_p=dropout_p,
                 device=device, 
                 lower=-1.0,upper=1.0)
        self.net_diffuse = MLP(n_input=n_constituent + n_coarse_code, 
                 n_hidden=n_hidden, 
                 n_output=3,
                dropout_p=dropout_p,
                 device=device, 
                 lower=-1.0,upper=1.0) 

        self.coarse_code_template = torch.ones((1,n_channel,n_coarse_code), dtype=torch.float32,device=device)
        sigma = 0.25
        const_1 = 1.0 / (sigma * np.sqrt(6.28)) # 2 * pi
        for i in range(n_channel):
            ii = i / (n_channel - 1)
            for j in range(n_coarse_code):
                jj = j / (n_coarse_code - 1)
                self.coarse_code_template[:,i,j] = const_1 * np.exp(-0.5 * np.square((ii - jj)/sigma))

    def reset_dropout(self,dropout_p):
        self.net_direct.reset_dropout(dropout_p)
        self.net_diffuse.reset_dropout(dropout_p)

    #@torch.compile
    def forward(self, x):

        tau, mu_direct, mu_diffuse, constituents = x

        tau_total = torch.sum(tau, dim=2, keepdims=False)

        t_direct = torch.exp(-tau_total / (mu_direct + eps_1))
        t_diffuse = torch.exp(-tau_total / (mu_diffuse + eps_1))

        constituents_direct = constituents / (mu_direct + eps_1)
        constituents_direct = torch.concat((constituents_direct, mu_direct),dim=1)

        constituents_diffuse = constituents 

        # Add channel dim to constituents
        constituents_direct = torch.unsqueeze(constituents_direct,dim=1)
        constituents_diffuse = torch.unsqueeze(constituents_diffuse,dim=1)

        constituents_direct = constituents_direct.repeat(1,self.n_channel,1)
        constituents_diffuse = constituents_diffuse.repeat(1,self.n_channel,1) 

        # Repeat coarse code over all inputs
        coarse_code_input = self.coarse_code_template.repeat(constituents.shape[0],1,1)

        # Concatenate coarse code onto inputs
        constituents_direct = torch.concat([constituents_direct,coarse_code_input],dim=2)
        constituents_diffuse = torch.concat([constituents_diffuse,coarse_code_input],dim=2)

        e_split_direct = F.softmax(self.net_direct(constituents_direct), dim=-1) 
        e_split_diffuse = F.softmax(self.net_diffuse(constituents_diffuse), dim=-1)

        layers = [t_direct, t_diffuse, e_split_direct, e_split_diffuse]

        return layers
    

class Scattering_v3(nn.Module):
    """ Computes full set of layer properties for
    direct and diffuse transmission, reflection,
    and absorption

    Uses a single net for scattering for all channels

    Same as v2 except uses tau as input instead of constituents (gas concentrations)
    Tau makes more sense since it is the extinction amount rather than
    the amount of constituent as in v2
    """
    def __init__(self, n_channel, n_constituent, n_coarse_code, dropout_p, device):
        super(Scattering_v3, self).__init__()
        self.n_channel = n_channel
        self.n_coarse_code = n_coarse_code
        #n_hidden = [10, 6, 6, 6, 6]
        n_hidden = [10, 8, 8, 8, 8, 8]
        """ Computes split of extinguished radiation into absorbed, diffuse transmitted, and
        diffuse reflected """
        self.net_direct = MLP(n_input=n_constituent + n_coarse_code + 1, 
                 n_hidden=n_hidden, 
                 n_output=3,
                dropout_p=dropout_p,
                 device=device, 
                 lower=-1.0,upper=1.0)
        self.net_diffuse = MLP(n_input=n_constituent + n_coarse_code, 
                 n_hidden=n_hidden, 
                 n_output=3,
                dropout_p=dropout_p,
                 device=device, 
                 lower=-1.0,upper=1.0) 

        self.coarse_code_template = torch.ones((1,n_channel,n_coarse_code), dtype=torch.float32,device=device)
        sigma = 0.25
        const_1 = 1.0 / (sigma * np.sqrt(6.28)) # 2 * pi
        for i in range(n_channel):
            ii = i / (n_channel - 1)
            for j in range(n_coarse_code):
                jj = j / (n_coarse_code - 1)
                self.coarse_code_template[:,i,j] = const_1 * np.exp(-0.5 * np.square((ii - jj)/sigma))

    def reset_dropout(self,dropout_p):
        self.net_direct.reset_dropout(dropout_p)
        self.net_diffuse.reset_dropout(dropout_p)

    #@torch.compile
    def forward(self, x):

        tau, mu_direct, mu_diffuse, _ = x

        tau_total = torch.sum(tau, dim=2, keepdims=False)

        t_direct = torch.exp(-tau_total / (mu_direct + eps_1))
        t_diffuse = torch.exp(-tau_total / (mu_diffuse + eps_1))

        mu_direct = torch.unsqueeze(mu_direct,dim=1)
        #mu_diffuse = torch.unsqueeze(mu_diffuse,dim=1)

        tau_direct = tau / (mu_direct + eps_1)
        mu_direct = mu_direct.repeat(1,self.n_channel,1)
        tau_direct = torch.concat((tau_direct, mu_direct),dim=2)

        tau_diffuse = tau 

        # Repeat coarse code over all inputs
        coarse_code_input = self.coarse_code_template.repeat(tau.shape[0],1,1)

        # Concatenate coarse code onto inputs
        tau_direct = torch.concat([tau_direct,coarse_code_input],dim=2)
        tau_diffuse = torch.concat([tau_diffuse,coarse_code_input],dim=2)

        e_split_direct = F.softmax(self.net_direct(tau_direct), dim=-1) 
        e_split_diffuse = F.softmax(self.net_diffuse(tau_diffuse), dim=-1)

        layers = [t_direct, t_diffuse, e_split_direct, e_split_diffuse]

        return layers


class Scattering_v3_tau(nn.Module):
    """ 
    Same as v3 except consolidates the tau's of (o3, c2o, n2o, ch4) into a single input

    Reduces number of inputs from 12 to 9
    Reduce first layer from 10 to 7 hidden units, reduced other
    hidden units from 8 to 7
    """
    def __init__(self, n_channel, n_constituent, n_coarse_code, dropout_p, device):
        super(Scattering_v3_tau, self).__init__()
        self.n_channel = n_channel
        self.n_coarse_code = n_coarse_code
        #n_hidden = [10, 6, 6, 6, 6]
        n_hidden = [10, 8, 8, 8, 8, 8]
        n_hidden = [7, 7, 7, 7, 7, 7]
        n_input = 5
        """ Computes split of extinguished radiation into absorbed, diffuse transmitted, and
        diffuse reflected """
        self.net_direct = MLP(n_input=n_input + n_coarse_code + 1, 
                 n_hidden=n_hidden, 
                 n_output=3,
                dropout_p=dropout_p,
                 device=device, 
                 lower=-1.0,upper=1.0)
        self.net_diffuse = MLP(n_input=n_input + n_coarse_code, 
                 n_hidden=n_hidden, 
                 n_output=3,
                dropout_p=dropout_p,
                 device=device, 
                 lower=-1.0,upper=1.0) 

        self.coarse_code_template = torch.ones((1,n_channel,n_coarse_code), dtype=torch.float32,device=device)
        sigma = 0.25
        const_1 = 1.0 / (sigma * np.sqrt(6.28)) # 2 * pi
        for i in range(n_channel):
            ii = i / (n_channel - 1)
            for j in range(n_coarse_code):
                jj = j / (n_coarse_code - 1)
                self.coarse_code_template[:,i,j] = const_1 * np.exp(-0.5 * np.square((ii - jj)/sigma))

    def reset_dropout(self,dropout_p):
        self.net_direct.reset_dropout(dropout_p)
        self.net_diffuse.reset_dropout(dropout_p)

    #@torch.compile
    def forward(self, x):

        tau, mu_direct, mu_diffuse, _ = x

        # sum over gases
        tau_total = torch.sum(tau, dim=2, keepdims=False)

        t_direct = torch.exp(-tau_total / (mu_direct + eps_1))
        t_diffuse = torch.exp(-tau_total / (mu_diffuse + eps_1))

        mu_direct = torch.unsqueeze(mu_direct,dim=1)
        tau_direct = tau / (mu_direct + eps_1)

        direct_1 = tau_direct[:,:,3] + tau_direct[:,:,4]+ tau_direct[:,:,6] + tau_direct[:,:,7]

        direct_1 = torch.unsqueeze(direct_1,dim=2)

        # Add channels
        mu_direct = mu_direct.repeat(1,self.n_channel,1)

        # lwp, iwp, dry gas, water vapor, (o3, c2o, n2o, ch4)
        direct = torch.concat((tau_direct[:,:,0:2], tau_direct[:,:,5:6],
                               tau_direct[:,:,2:3], direct_1, mu_direct),dim=2)

        diffuse_1 = tau[:,:,3] + tau[:,:,4]+ tau[:,:,6] + tau[:,:,7]

        diffuse_1 = torch.unsqueeze(diffuse_1,dim=2)

        diffuse = torch.concat((tau[:,:,0:2], tau[:,:,5:6], 
                                tau[:,:,2:3], diffuse_1), dim=2)

        # Repeat coarse code over all inputs
        coarse_code_input = self.coarse_code_template.repeat(tau.shape[0],1,1)

        # Concatenate coarse code onto inputs
        direct = torch.concat([direct,coarse_code_input],dim=2)
        diffuse = torch.concat([diffuse,coarse_code_input],dim=2)

        e_split_direct = F.softmax(self.net_direct(direct), dim=-1) 
        e_split_diffuse = F.softmax(self.net_diffuse(diffuse), dim=-1)

        layers = [t_direct, t_diffuse, e_split_direct, e_split_diffuse]

        return layers
class MultiReflection(nn.Module):
    """ 
    Recomputes each layer's radiative coefficients by accounting
    for interaction (multireflection) with all other layers using the 
    Adding-Doubling method (no learning).
    """

    def __init__(self):
        super(MultiReflection, self).__init__()

    def _adding_doubling (self, t_direct, t_diffuse, 
                        e_split_direct, e_split_diffuse, 
                        r_surface_direct, r_surface_diffuse, 
                        a_surface_direct, a_surface_diffuse):
        """
        Multireflection between a single layer and a surface using the
        Adding-Doubling Method.

        See p.418-424 of "A First Course in Atmospheric Radiation (2nd edition)"
        by Grant W. Petty
        Also see Shonk and Hogan, 2007

        Input and Output Shape:
            (n_samples, n_channels, . . .)

        Arguments:

            t_direct, t_diffuse - Direct transmission coefficients of 
                the layer.  
                - These are not changed by multi reflection
                - t_diffuse is for diffuse input that is directly 
                transmitted.

            e_split_direct, e_split_diffuse - The layer's split of extinguised  
                radiation into transmitted, reflected,
                and absorbed components. These components 
                sum to 1.0. The transmitted and reflected components produce
                diffuse radiation.
                
            r_surface_direct, r_surface_diffuse - The original reflection 
                coefficients of the surface.

            a_surface_direct, a_surface_diffuse - The original absorption 
                coefficients of the surface. 
                
        Returns:

            t_multi_direct, t_multi_diffuse - The layer's transmission
                coefficients for radiation that is multi-reflected (as 
                opposed to directly transmitted, e.g., t_direct, t_diffuse)

            r_layer_multi_direct, r_layer_multi_diffuse - The layer's 
                reflection coefficients after accounting for multi-reflection 
                with the surface

            r_surface_multi_direct, r_surface_multi_diffuse - The surface's
                reflection coefficients after accounting for 
                multi-reflection with the layer

            a_layer_multi_direct, a_layer_multi_diffuse - The layer's 
                absorption coefficients layer after accounting for 
                multi-reflection with surface

            a_surface_multi_direct, a_surface_multi_diffuse - The surface's
                absorption coefficients after accounting for multi-reflection 
                with the layer

        Notes:
        
            Conservation of energy:

                1.0 = a_surface_direct + r_surface_direct
                1.0 = a_surface_diffuse + r_surface_diffuse

                1.0 = a_surface_multi_direct + a_layer_multi_direct + 
                        r_layer_multi_direct
                1.0 = a_surface_multi_diffuse + a_layer_multi_diffuse + 
                        r_layer_multi_diffuse

                The absorption at the layer (after accounting for 
                multi-reflection) must equal the combined loss of flux for 
                the downward and upward streams:
            
                a_layer_multi_direct = (1 - t_direct - t_multi_direct) + 
                                (r_surface_multi_direct - r_layer_multi_direct)
                a_layer_multi_diffuse = (1 - t_diffuse - t_multi_diffuse) + 
                            (r_surface_multi_diffuse - r_layer_multi_diffuse)

            When merging the multireflected layer and the surface into 
            a new "surface", the reflection coefficient is just the reflection
            of the layer. However, the absorption of the new surface
            is the sum of the surface and layer absorptions:

                r_layer_multi_direct => r_surface_direct
                a_layer_multi_direct + a_surface_multi_direct => 
                                                            a_surface_direct

            See class Propagation below for how the multi-reflection
            coefficients are used to propagate radiation 
            downward from the top of the atmosphere
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

    def forward(self, x):
        """
        Multi reflects between a surface and a single 
        overhead layer generating their revised radiative coefficients.
        Then merges this surface and layer into a new
        "surface" and we repeat this process with 
        the next layer above and continues
        to the top of the atmosphere (TOA).

        Computations are independent across channel.

        The prefixes -- t, e, r, a -- correspond respectively to
        transmission, extinction, reflection, and absorption.
        """

        radiative_layers, x_surface = x

        t_direct, t_diffuse, e_split_direct, e_split_diffuse = radiative_layers

        (r_surface_direct, r_surface_diffuse, 
         a_surface_direct, a_surface_diffuse) = (x_surface[:,:,0], 
                                               x_surface[:,:,1], 
                                               x_surface[:,:,2], 
                                               x_surface[:,:,3])
        t_multi_direct_list = []
        t_multi_diffuse_list = []
        r_surface_multi_direct_list = []
        r_surface_multi_diffuse_list = []
        a_layer_multi_direct_list = []
        a_layer_multi_diffuse_list = []

        # Start at the original surface and the first layer and move up
        # one layer for each iteration
        for i in reversed(torch.arange(start=0, end=t_direct.shape[1])):
            multireflected_info = self._adding_doubling (t_direct[:,i,:], 
                                                   t_diffuse[:,i,:], 
                                                   e_split_direct[:,i,:,:], 
                                                   e_split_diffuse[:,i,:,:], 
                                                   r_surface_direct, 
                                                   r_surface_diffuse, 
                                                   a_surface_direct, 
                                                   a_surface_diffuse)
            (t_multi_direct, t_multi_diffuse,
            r_layer_multi_direct, r_layer_multi_diffuse,
            r_surface_multi_direct, r_surface_multi_diffuse,
            a_layer_multi_direct, a_layer_multi_diffuse,
            a_surface_multi_direct, a_surface_multi_diffuse) = multireflected_info

            # Merge the layer and surface forming a new "surface"
            r_surface_direct = r_layer_multi_direct
            r_surface_diffuse = r_layer_multi_diffuse
            a_surface_direct = a_layer_multi_direct + a_surface_multi_direct
            a_surface_diffuse = a_layer_multi_diffuse + a_surface_multi_diffuse

            t_multi_direct_list.append(t_multi_direct)
            t_multi_diffuse_list.append(t_multi_diffuse)
            r_surface_multi_direct_list.append(r_surface_multi_direct)
            r_surface_multi_diffuse_list.append(r_surface_multi_diffuse)
            a_layer_multi_direct_list.append(a_layer_multi_direct)
            a_layer_multi_diffuse_list.append(a_layer_multi_diffuse)

        # Stack output in layers
        t_multi_direct= torch.stack(t_multi_direct_list, dim=1)
        t_multi_diffuse = torch.stack(t_multi_diffuse_list, dim=1)
        r_surface_multi_direct = torch.stack(r_surface_multi_direct_list, dim=1)
        r_surface_multi_diffuse = torch.stack(r_surface_multi_diffuse_list, dim=1)
        a_layer_multi_direct = torch.stack(a_layer_multi_direct_list, dim=1)
        a_layer_multi_diffuse = torch.stack(a_layer_multi_diffuse_list, dim=1)

        # Reverse ordering of layers such that top layer is first
        t_multi_direct = torch.flip(t_multi_direct, dims=(1,))
        t_multi_diffuse = torch.flip(t_multi_diffuse, dims=(1,))
        r_surface_multi_direct = torch.flip(r_surface_multi_direct, dims=(1,))
        r_surface_multi_diffuse = torch.flip(r_surface_multi_diffuse, dims=(1,))
        a_layer_multi_direct = torch.flip(a_layer_multi_direct, dims=(1,))
        a_layer_multi_diffuse = torch.flip(a_layer_multi_diffuse, dims=(1,))

        multireflected_layers = [t_direct, t_diffuse, 
                                 t_multi_direct, t_multi_diffuse, 
                                 r_surface_multi_direct,r_surface_multi_diffuse, 
                                 a_layer_multi_direct, a_layer_multi_diffuse]
        # The reflection coefficient at the top of the atmosphere
        # is the reflection coefficient of top layer
        upward_reflection_toa = r_layer_multi_direct
        return (multireflected_layers, upward_reflection_toa)
    
class Propagation(nn.Module):
    """
    Propagate flux from the top of the atmosphere to the
    surface.
    We only need to propagate flux in a single pass
    since the radiative properties account for
    multi reflection

    Consider two downward fluxes entering the layer: 
                flux_direct, flux_diffuse

    Downward Direct Flux Transmitted = flux_direct * t_direct
    Downward Diffuse Flux Transmitted = 
                    flux_direct * t_multi_direct + 
                    flux_diffuse * (t_diffuse + t_multi_diffuse)

    Upward Flux from Top Layer = flux_direct * r_layer_multi_direct +
                            flux_diffuse * r_layer_multi_diffuse

    Upward Flux into Top Layer = 
                        flux_direct * r_surface_multi_direct +
                        flux_diffuse * r_surface_multi_diffuse

    Both upward fluxes are diffuse since they are from radiation
    that is scattered upwards
    """
    def __init__(self,n_channel):
        super().__init__()
        super(Propagation, self).__init__()
        self.n_channel = n_channel

    def forward(self, x):

        multireflected_layers, upward_reflection_toa, input_flux = x

        (t_direct, t_diffuse,
        t_multi_direct, t_multi_diffuse,
        r_surface_multi_direct, r_surface_multi_diffuse,
        a_layer_multi_direct, a_layer_multi_diffuse)  = multireflected_layers

        flux_direct, flux_diffuse = input_flux

        # Assign all 3 fluxes above the top layer
        flux_down_direct = [flux_direct]
        flux_down_diffuse = [flux_diffuse]
        flux_up_diffuse = [flux_direct * upward_reflection_toa]

        flux_absorbed = []

        for i in range(t_direct.shape[1]):

            flux_absorbed.append(
                flux_direct * a_layer_multi_direct[:,i]  
                + flux_diffuse * a_layer_multi_diffuse[:,i])

            # Will want this later when incorporate surface interactions
            #flux_absorbed_surface = flux_direct * a_surface_multi_direct + \
            #flux_diffuse * a_surface_multi_diffuse

            flux_down_direct.append(flux_direct * t_direct[:,i])
            flux_down_diffuse.append(
                flux_direct * t_multi_direct[:,i] 
                + flux_diffuse * (t_diffuse[:,i] + t_multi_diffuse[:,i]))
            
            flux_up_diffuse.append(
                flux_direct * r_surface_multi_direct[:,i] 
                + flux_diffuse * r_surface_multi_diffuse[:,i])
            
            flux_direct = flux_down_direct[-1]
            flux_diffuse = flux_down_diffuse[-1]
        
        flux_down_direct = torch.stack(flux_down_direct,dim=1)
        flux_down_diffuse = torch.stack(flux_down_diffuse,dim=1)
        flux_up_diffuse = torch.stack(flux_up_diffuse,dim=1)
        flux_absorbed = torch.stack(flux_absorbed,dim=1)

        # Sum across channels
        flux_down_direct = torch.sum(flux_down_direct,dim=2,keepdim=False)
        flux_down_diffuse = torch.sum(flux_down_diffuse,dim=2,keepdim=False)      
        flux_up_diffuse = torch.sum(flux_up_diffuse,dim=2,keepdim=False)  
        flux_absorbed = torch.sum(flux_absorbed,dim=2,keepdim=False)  

        return [flux_down_direct, flux_down_diffuse, flux_up_diffuse, 
                flux_absorbed]

class DownwardPropagationDirect(nn.Module):
    """    
    Propagate direct downward flux from the top of the atmosphere 
    to the surface.
    """
    def __init__(self,n_channel):
        super(DownwardPropagationDirect, self).__init__()
        self.n_channel = n_channel

    def forward(self, x):
        input_flux, t_direct = x
        flux_down_direct = [input_flux]
        for i in range(t_direct.size(1)):
            output_flux = input_flux * t_direct[:,i,:]
            flux_down_direct.append(output_flux)
            input_flux = output_flux
        flux_down_direct = torch.stack(flux_down_direct,dim=1)
        flux_down_direct = torch.sum(flux_down_direct,dim=2,keepdim=False)
        return flux_down_direct

class DirectDownwardNet(nn.Module):
    """ Computes radiative transfer for downward direct radiation only """
    def __init__(self, n_channel, n_constituent, device):
        super(DirectDownwardNet, self).__init__()
        self.device = device
        self.n_constituent = n_constituent

        # Learns decompositon of input solar radiation into channels
        self.spectral_net = nn.Linear(1,n_channel,bias=False,device=device)
        torch.nn.init.uniform_(self.spectral_net.weight, a=0.4, b=0.6)

        # Learns optical depth for each constituent for each channel
        self.extinction_net = LayerDistributed(Extinction(n_channel,device))

        # Computes direct transmission coefficient for each channel
        self.direct_transmission_net = LayerDistributed(DirectTransmission())

        # Progates radiation from top of atmosphere (TOA) to surface
        self.downward_propagate = DownwardPropagationDirect(n_channel)

    def forward(self, x):
        x_layers, _, _, _ = x
        (mu_direct, temperature_pressure_log_pressure, 
         constituents) = (x_layers[:,:,0:1], x_layers[:,:,1:4], 
                          x_layers[:,:,4:4+self.n_constituent])

        one = torch.unsqueeze(
            torch.ones((mu_direct.shape[0]),
                       dtype=torch.float32,device=self.device), 1)
        
        input_flux = F.softmax(self.spectral_net(one), dim=-1)

        tau = self.extinction_net((temperature_pressure_log_pressure, constituents))

        t_direct = self.direct_transmission_net((mu_direct, tau))

        flux_down_direct = self.downward_propagate((input_flux,t_direct))

        return flux_down_direct
    
class FullNet(nn.Module):
    """ Computes full radiative transfer (direct and diffuse radiation)
    for an atmospheric column """

    def __init__(self, n_channel, n_constituent, dropout_p, device):
        super(FullNet, self).__init__()
        self.device = device
        self.n_channel = n_channel
        n_coarse_code = 3

        # Learns single diffuse zenith angle approximation 
        self.mu_diffuse_net = nn.Linear(1,1,bias=False,device=device)
        torch.nn.init.uniform_(self.mu_diffuse_net.weight, a=0.4, b=0.6)

        # Learns decompositon of input solar radiation into channels
        self.spectral_net = nn.Linear(1,n_channel,bias=False,device=device)
        torch.nn.init.uniform_(self.spectral_net.weight, a=0.4, b=0.6)

        # Learns optical depth for each layer for each constituent for 
        # each channel
        self.extinction_net = LayerDistributed(Extinction(n_channel,dropout_p,
                                                          device))

        # Learns decomposition of extinguished radiation (into t, r, a)
        # for each channel
        if False:
            self.scattering_net = LayerDistributed(Scattering_v1_tau(n_channel,
                                                          n_constituent,
                                                          dropout_p,
                                                          device))

        elif False:
                # Corresponds to Torch.Dataloader.2a for starting 
                # epoch (selects #6) and Torch.Dataloader.2 for remaining epoches
                self.scattering_net = LayerDistributed(Scattering_v3_tau(n_channel,
                                    n_constituent,
                                                          n_coarse_code,
                                                          dropout_p,
                                                          device))
        else:
            self.scattering_net = LayerDistributed(Scattering_v3(n_channel,
                                                          n_constituent,
                                                          n_coarse_code,
                                                          dropout_p,
                                                          device))
        # Computes result of interaction among all layers
        self.multireflection_net = MultiReflection()

        # Propagates radiation from top of atmosphere (TOA) to surface
        self.propagation_net = Propagation(n_channel)


    def reset_dropout(self,dropout_p):
        self.extinction_net.reset_dropout(dropout_p)
        self.scattering_net.reset_dropout(dropout_p)

    def forward(self, x):

        x_layers, x_surface, _, _, _, _ = x

        #print(f"x_layers.shape = {x_layers.shape}")
        (mu_direct, 
        temperature_pressure_log_pressure, 
        constituents) = (x_layers[:,:,0:1], 
                        x_layers[:,:,1:4], 
                        x_layers[:,:,4:12])

        one = torch.ones((mu_direct.shape[0],1),dtype=torch.float32,
                        device=self.device)
        mu_diffuse = torch.sigmoid(self.mu_diffuse_net(one))
        mu_diffuse = mu_diffuse.repeat([1,mu_direct.shape[1]])
        mu_diffuse = torch.unsqueeze(mu_diffuse,dim=2)

        tau = self.extinction_net((temperature_pressure_log_pressure, 
                                constituents))
        
        #print(f"First: tau.shape = {tau.shape}")

        layers = self.scattering_net((tau, mu_direct, mu_diffuse, 
                                                constituents))

        (multireflected_layers, 
        upward_reflection_toa) = self.multireflection_net([layers,
                                                        x_surface])

        flux_direct = F.softmax(self.spectral_net(one),dim=-1)
        flux_diffuse = torch.zeros((mu_direct.shape[0], self.n_channel),
                                        dtype=torch.float32,
                                        device=self.device)
        input_flux = [flux_direct, flux_diffuse]

        flux = self.propagation_net((multireflected_layers, 
                                    upward_reflection_toa,
                                    input_flux))

        (flux_down_direct, flux_down_diffuse, flux_up_diffuse, 
        flux_absorbed) = flux
        
        flux_down = flux_down_direct + flux_down_diffuse
        flux_up = flux_up_diffuse

        return [flux_down_direct, flux_down, flux_up, flux_absorbed]

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

def loss_heating_rate_direct_full_wrapper(data, y_pred, weight_profile):
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

def loss_heating_rate_full_wrapper(data, y_pred, weight_profile):
    _, _, toa, delta_pressure, _, flux_absorbed_true = data
    _, _, _, flux_absorbed_pred = y_pred
    loss = loss_heating_rate_full(flux_absorbed_true, flux_absorbed_pred, 
                                  toa, delta_pressure)
    return loss

def loss_ukkonen_direct(y_true, y_pred, toa, delta_pressure, weight_profile):
    loss_flux = loss_weighted (y_true, y_pred, weight_profile)
    hr_loss = loss_heating_rate_direct(y_true, y_pred, toa, delta_pressure)
    alpha   = 1.0e-4
    return alpha * hr_loss + (1.0 - alpha) * loss_flux

def loss_ukkonen_direct_wrapper(data, y_pred, weight_profile):
    _, y_true, x_toa, x_delta_pressure = data
    loss = loss_ukkonen_direct(y_true,y_pred,x_toa,x_delta_pressure,weight_profile)
    return loss

def loss_flux_direct_wrapper(data, y_pred, weight_profile):
    _, y_true, _, _ = data
    loss = loss_weighted(y_true,y_pred,weight_profile)
    return loss

def loss_flux_full_wrapper(data, y_pred, weight_profile):
    _, _, toa, _, y_true, _ = data
    flux_down_true, flux_up_true = y_true[:,:,1], y_true[:,:,2]
    _, flux_down_pred, flux_up_pred, _ = y_pred
    
    flux_pred = torch.concat((flux_down_pred,flux_up_pred),dim=1)
    flux_true = torch.concat((flux_down_true,flux_up_true),dim=1)
    loss = loss_weighted(flux_true, flux_pred, toa)
    return loss

def loss_flux_direct_wrapper_2(data, y_pred, weight_profile):
    _, _, toa, _, y_true, _ = data
    flux_down_direct_pred, _, _, _ = y_pred
    flux_down_direct_true = y_true[:,:,0]
    
    loss = loss_weighted(flux_down_direct_true, flux_down_direct_pred, toa)
    return loss

def loss_flux_full_wrapper_2(data, y_pred, weight_profile):
    _, _, _, _, y_true, _ = data
    flux_down_true, flux_up_true = y_true[:,:,1], y_true[:,:,2]
    _, flux_down_pred, flux_up_pred, _ = y_pred
    
    flux_pred = torch.concat((flux_down_pred,flux_up_pred),dim=1)
    flux_true = torch.concat((flux_down_true,flux_up_true),dim=1)
    loss = loss_weighted(flux_true, flux_pred, weight_profile)
    return loss

def loss_henry(y_true, flux_absorbed_true, y_pred, toa_weighting_profile, 
               delta_pressure, weight_profile):
    
    (flux_down_direct_pred, flux_down_pred, 
     flux_up_pred, flux_absorbed_pred) = y_pred
    
    (flux_down_direct_true, flux_down_true, 
     flux_up_true) = (y_true[:,:,0], y_true[:,:,1], y_true[:,:,2])
    
    flux_pred = torch.concat((flux_down_pred,flux_up_pred),dim=1)
    flux_true = torch.concat((flux_down_true,flux_up_true),dim=1)

    flux_loss = loss_weighted(flux_true, flux_pred, weight_profile)
    hr_loss = loss_heating_rate_full(flux_absorbed_true, flux_absorbed_pred,
                                     toa_weighting_profile, delta_pressure)
    hr_direct_loss = loss_heating_rate_direct(
        flux_down_direct_true, flux_down_direct_pred, 
        toa_weighting_profile, delta_pressure)
    
    alpha   = 1.0e-4
    return alpha * (hr_loss + hr_direct_loss) + (1.0 - alpha) * flux_loss

def loss_henry_2(y_true, flux_absorbed_true, y_pred, toa_weighting_profile, 
               delta_pressure, weight_profile):
    # Handles flux using TOA weight rather than weight profile
    
    (flux_down_direct_pred, flux_down_pred, 
     flux_up_pred, flux_absorbed_pred) = y_pred
    
    (flux_down_direct_true, flux_down_true, 
     flux_up_true) = (y_true[:,:,0], y_true[:,:,1], y_true[:,:,2])
    
    flux_pred = torch.concat((flux_down_pred,flux_up_pred),dim=1)
    flux_true = torch.concat((flux_down_true,flux_up_true),dim=1)

    flux_loss = loss_weighted(flux_true, flux_pred, toa_weighting_profile)
    flux_direct_loss = loss_weighted(flux_down_direct_true, flux_down_direct_pred, toa_weighting_profile)

    hr_loss = loss_heating_rate_full(flux_absorbed_true, flux_absorbed_pred,
                                     toa_weighting_profile, delta_pressure)
    
    hr_direct_loss = loss_heating_rate_direct(
        flux_down_direct_true, flux_down_direct_pred, 
        toa_weighting_profile, delta_pressure)
    


    if False:
        alpha   = 0.3
        return alpha * (hr_loss + 2.0 * hr_direct_loss) + (1.0 - alpha) * (flux_loss + 2.0 * flux_direct_loss)
    else:
        alpha   = 0.66667
        return alpha * (hr_loss + hr_direct_loss) + (1.0 - alpha) * (flux_loss + flux_direct_loss)

def loss_henry_full_wrapper(data, y_pred, weight_profile):
    _, _, toa, delta_pressure, y_true, flux_absorbed_true = data
    loss = loss_henry_2(y_true, flux_absorbed_true, y_pred, toa, 
               delta_pressure, weight_profile)
    return loss


def train_loop(dataloader, model, optimizer, loss_function, weight_profile, device):
    """ Generic training loop """

    model.train()

    loss_string = "Training Loss: "
    for batch, data in enumerate(dataloader):
        data = [x.to(device) for x in data]
        y_pred = model(data)
        loss = loss_function(data, y_pred, weight_profile)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 20 == 0:
            loss_value = loss.item()
            loss_string += f" {loss_value:.9f}"

    #print (loss_string)

def test_loop(dataloader, model, loss_functions, loss_names, weight_profile, device):
    """ Generic testing / evaluation loop """
    model.eval()
    num_batches = len(dataloader)

    loss = np.zeros(len(loss_functions), dtype=np.float32)

    with torch.no_grad():
        for data in dataloader:
            data = [x.to(device) for x in data]
            y_pred = model(data)
            for i, loss_fn in enumerate(loss_functions):
                loss[i] += loss_fn(data, y_pred, weight_profile).item()

    loss /= num_batches

    print(f"Test Error: ")
    for i, value in enumerate(loss):
        print(f" {loss_names[i]}: {value:.8f}")
    print("")

    return loss

def tensorize(np_ndarray):
    t = torch.from_numpy(np_ndarray).float()
    return t

def train_direct_only():

    print("Pytorch version:", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    datadir     = "/data-T1/hws/tmp/"
    filename_training = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"
    filename_validation = datadir + "/RADSCHEME_data_g224_CAMS_2014.2.nc"
    filename_testing = datadir + "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"
    filename_direct_model = datadir + "/Direct_Torch.3."

    batch_size = 2048
    n_channel = 30
    n_constituent = 8
    model = DirectDownwardNet(n_channel,n_constituent,device)
    model = model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters())

    checkpoint_period = 100
    epochs = 4000

    x_layers, y_true, x_toa, x_delta_pressure = load_data_direct_pytorch(filename_training, n_channel)

    weight_profile = 1.0 / torch.mean(tensorize(y_true), dim=0, keepdim=True)


    train_dataset = torch.utils.data.TensorDataset(tensorize(x_layers), 
                                                   tensorize(y_true),
                                                   tensorize(x_toa),
                                                   tensorize(x_delta_pressure))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

    x_layers, y_true, x_toa, x_delta_pressure = load_data_direct_pytorch(filename_validation, n_channel)
    validation_dataset = torch.utils.data.TensorDataset(tensorize(x_layers), 
                                                   tensorize(y_true),
                                                   tensorize(x_toa),
                                                   tensorize(x_delta_pressure))

    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size, shuffle=True)

    loss_functions = (loss_ukkonen_direct_wrapper, loss_heating_rate_direct_wrapper,
                      loss_flux_direct_wrapper)
    
    loss_names = ("Loss", "Direct Heating Rate Loss", "Flux Loss")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    if True:
        t = 0
    else:   
        t = 100
        checkpoint = torch.load(filename_direct_model + str(t))
        print(f"Loaded Model: epoch = {t}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #epoch = checkpoint['epoch']
    while t < epochs:
        t += 1
        print(f"Epoch {t}\n-------------------------------")
        #with profiler.profile(with_stack=True, profile_memory=True) as prof:
        start.record()
        train_loop(train_dataloader, model, optimizer, loss_ukkonen_direct_wrapper,weight_profile, device)
        loss = test_loop(validation_dataloader, model, loss_functions, loss_names, weight_profile, device)
        end.record()
        torch.cuda.synchronize()
        print(f" Elapsed time in seconds: {start.elapsed_time(end) / 1000.0}\n")

        if t % checkpoint_period == 0:
            torch.save({
            'epoch': t,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, filename_direct_model + str(t))
            print(f' Wrote Model: epoch = {t}')

    print("Done!")
    #print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

def get_weight_profile(y_flux,device):
    rsd = torch.from_numpy(y_flux[:,:,1]).float().to(device)
    rsu = torch.from_numpy(y_flux[:,:,2]).float().to(device)
    flux = torch.concat((rsd,rsu),dim=1)
    weight_profile = 1.0 / torch.mean(flux,dim=0,keepdim=True)
    return weight_profile

def get_weight_profile_2(device):
    import xarray as xr
    datadir     = "/data-T1/hws/tmp/"
    filename_training = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"
    dt = xr.open_dataset(filename_training)
    rsd = dt['rsd'].data
    rsu = dt['rsu'].data
    shape = rsd.shape
    rsd = np.reshape(rsd, (shape[0]*shape[1], shape[2]))
    rsu = np.reshape(rsu, (shape[0]*shape[1], shape[2]))
    toa = np.copy(rsd[:,0:1])
    rsu = rsu / toa
    rsd = rsd / toa
    rsd = torch.from_numpy(rsd).float().to(device)
    rsu = torch.from_numpy(rsu).float().to(device)
    flux = torch.concat((rsd,rsu),dim=1)
    #weight_profile = 1.0 / torch.mean(flux,dim=0,keepdim=True)
    #weight_profile = 1.0 / torch.mean(flux,dim=(0,1),keepdim=True)
    weight_profile = 3.0 / torch.mean(flux,dim=(0,1),keepdim=True)
    dt.close()
    return weight_profile

def train_full():

    print("Pytorch version:", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    if torch.cuda.is_available():
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:',torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
        use_cuda = True
    else:
        use_cuda = False

    datadir     = "/data-T1/hws/tmp/"
    filename_training = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"
    filename_validation = datadir + "/RADSCHEME_data_g224_CAMS_2014.2.nc"
    filename_testing = datadir + "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"
    #filename_full_model = datadir + "/Full_Torch.3." # Uses Scattering_V2
    filename_full_model = datadir + "/Full_Torch.4." # Uses Scattering_V3 and tau input
    filename_full_model = datadir + "/Full_Torch.5." # Uses Scattering_V3 and tau input and extra layer
    filename_full_model = datadir + "/Full_Torch.6." # Same as #5 with width of 8 and additional extra layer

    batch_size = 2048
    n_channel = 30
    n_constituent = 8
    checkpoint_period = 50
    epochs = 4000
    t_start = 0
    t_warmup = 1
    t = t_start

    #dropout_schedule = (0.0, 0.21, 0.15, 0.1, 0.07, 0.0, 0.0) # 400
    #dropout_epochs = (-1, 50, 200, 300, 400, 500, epochs + 1)

    dropout_schedule = (0.0, 0.07, 0.1, 0.15, 0.2, 0.15, 0.1, 0.07, 0.0, 0.0) # 400
    dropout_epochs =   (-1, 200,   300, 350,  400, 450,  550, 650, 750, epochs + 1)

    # "/Full_Torch.4."
    #dropout_schedule = (0.0, 0.07, 0.1, 0.15, 0.2, 0.15, 0.1, 0.07, 0.0, 0.0) # 400
    #dropout_epochs =   (-1, 600,   700, 750,  800, 850,  950, 1050, 1150, epochs + 1)

    dropout_index = next(i for i, x in enumerate(dropout_epochs) if t <= x) - 1
    dropout_p = dropout_schedule[dropout_index]
    last_dropout_index = dropout_index


    model = FullNet(n_channel,n_constituent,dropout_p,device).to(device=device)
    optimizer = torch.optim.Adam(model.parameters())

    (x_layers, x_surface, x_toa, x_delta_pressure, 
     y_flux, y_flux_absorbed) = load_data_full_pytorch_2(filename_training, 
                                                         n_channel)

    weight_profile = get_weight_profile(y_flux, device)

    train_dataset = torch.utils.data.TensorDataset(tensorize(x_layers), 
                                                   tensorize(x_surface), 
                                                   tensorize(x_toa), 
                                                   tensorize(x_delta_pressure), 
                                                   tensorize(y_flux), 
                                                   tensorize(y_flux_absorbed))


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, 
                                                   shuffle=True,
                                                   pin_memory=True,
                                                        num_workers=1)

    (x_layers, x_surface, x_toa, x_delta_pressure, 
     y_flux, y_flux_absorbed) = load_data_full_pytorch_2(filename_validation, 
                                                         n_channel)
    
    validation_dataset = torch.utils.data.TensorDataset(tensorize(x_layers),
                                                        tensorize(x_surface), 
                                                        tensorize(x_toa), 
                                                        tensorize(x_delta_pressure), 
                                                        tensorize(y_flux), 
                                                        tensorize(y_flux_absorbed))

    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, 
                                                        batch_size, 
                                                        shuffle=False, 
                                                        pin_memory=True,
                                                        num_workers=1)
    
    #start = torch.cuda.Event(enable_timing=True)
    #end = torch.cuda.Event(enable_timing=True)

    loss_functions = (loss_henry_full_wrapper, loss_flux_full_wrapper_2, loss_heating_rate_full_wrapper,
                      loss_heating_rate_direct_full_wrapper, loss_flux_full_wrapper,
                      )
    loss_names = ("Loss", "Flux Loss (weight profile)", "Heating Rate Loss", 
                  "Direct Heating Rate Loss", 
                    "Flux Loss (TOA weighting)")

    if t > 0:
        checkpoint = torch.load(filename_full_model + str(t))
        print(f"Loaded Model: epoch = {t}")
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

        if t < t_start + t_warmup:
            train_loop(train_dataloader, model, optimizer, loss_henry_full_wrapper, weight_profile, device)

            loss = test_loop(validation_dataloader, model, loss_functions, loss_names,weight_profile, device)
        else:

            #with profile(
            #    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            #    with_stack=True, with_modules=True,
            #) as prof:
            #start.record()
            train_loop(train_dataloader, model, optimizer, loss_henry_full_wrapper, weight_profile, device)

            loss = test_loop(validation_dataloader, model, loss_functions, loss_names,weight_profile, device)

            #if use_cuda: 
                #torch.cuda.synchronize()
            #end.record()

            #print(f"\n Elapsed time in seconds: {start.elapsed_time(end) / 1000.0}\n")

            #print(prof.key_averages(group_by_stack_n=6).table(sort_by='self_cpu_time_total', row_limit=15))
        if t % checkpoint_period == 0:
            torch.save({
            'epoch': t,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, filename_full_model + str(t))
            print(f' Wrote Model: epoch = {t}')

    print("Done!")


def train_full_dataloader():

    print("Pytorch version:", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    if torch.cuda.is_available():
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:',torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
        use_cuda = True
    else:
        use_cuda = False

    datadir     = "/data-T1/hws/tmp/"
    train_input_dir = "/data-T1/hws/CAMS/processed_data/training/2008/"
    cross_input_dir = "/data-T1/hws/CAMS/processed_data/cross_validation/2008/"
    months = [str(m).zfill(2) for m in range(1,13)]
    train_input_files = [f'{train_input_dir}Flux_sw-2008-{month}.nc' for month in months]
    cross_input_files = [f'{cross_input_dir}Flux_sw-2008-{month}.nc' for month in months]

    batch_size = 2048
    n_channel = 30
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

    filename_full_model = datadir + f"/Torch.Dataloader.v2_5." # scattering_v3
    # Uses loss_henry_2 with 0.3 weight on heating and 0.7 on flux
    # log_v5-i.txt - initialization
    # log_v5 
    # log_v5_results

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

        if True:
            #initial_model_n = 0  #4
            initial_model_n = 5   #5
            t_start = 1
            filename_full_model_input = f'{filename_full_model}i' + str(initial_model_n).zfill(2)
        else:
            t_start = 385
            filename_full_model_input = filename_full_model + str(t_start).zfill(3)


    for ee in range(number_of_tries):

        #print(f"Model = {str(ee).zfill(3)}")


        #filename_full_model = filename_full_model_input  #

        t_warmup = 1  # for profiling
        t = t_start
        dropout_p = 0.00

        dropout_schedule = (0.0, 0.07, 0.1, 0.15, 0.2, 0.15, 0.1, 0.07, 0.0, 0.0) # 400
        #dropout_epochs =   (-1, 200,   300, 350,  400, 450,  550, 650, 750, epochs + 1)
        dropout_epochs =   (-1, 40,   60, 70,  80, 90,  110, 130, 150, epochs + 1)

        dropout_index = next(i for i, x in enumerate(dropout_epochs) if t <= x) - 1
        dropout_p = dropout_schedule[dropout_index]
        last_dropout_index = dropout_index

        model = FullNet(n_channel,n_constituent,dropout_p,device).to(device=device)
        optimizer = torch.optim.Adam(model.parameters())

        weight_profile = get_weight_profile_2(device)

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
                        

        loss_functions = (loss_henry_full_wrapper, loss_flux_full_wrapper, loss_flux_direct_wrapper_2, loss_heating_rate_full_wrapper,
                        loss_heating_rate_direct_full_wrapper, 
                        )
        loss_names = ("Loss", "Flux Loss", "Flux Loss Direct", "Heating Rate Loss", 
                    "Heating Rate Loss Direct", 
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

            if t < t_start + t_warmup:
                train_loop(train_dataloader, model, optimizer, loss_henry_full_wrapper, weight_profile, device)

                loss = test_loop(validation_dataloader, model, loss_functions, loss_names,weight_profile, device)
            else:

                #with profile(
                #    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                #    with_stack=True, with_modules=True,
                #) as prof:
                #start.record()
                train_loop(train_dataloader, model, optimizer, loss_henry_full_wrapper, weight_profile, device)

                loss = test_loop(validation_dataloader, model, loss_functions, loss_names,weight_profile, device)

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

def test_full():

    print("Pytorch version:", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    datadir     = "/data-T1/hws/tmp/"

    filename_testing = datadir + "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"
    filename_full_model = datadir + "/Full_Torch.6."

    batch_size = 2048
    n_channel = 30
    n_constituent = 8

    (x_layers, x_surface, x_toa, x_delta_pressure, 
     y_flux, y_flux_absorbed) = load_data_full_pytorch_2(filename_testing, n_channel)
    weight_profile = get_weight_profile(y_flux, device)
    test_dataset = torch.utils.data.TensorDataset(tensorize(x_layers), 
                                                   tensorize(x_surface), 
                                                   tensorize(x_toa), 
                                                   tensorize(x_delta_pressure), 
                                                   tensorize(y_flux), 
                                                   tensorize(y_flux_absorbed))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True)

    model = FullNet(n_channel,n_constituent,dropout_p=0,device=device)
    model = model.to(device=device)

    loss_functions = (loss_henry_full_wrapper, loss_heating_rate_full_wrapper,
                      loss_heating_rate_direct_full_wrapper, loss_flux_full_wrapper)
    loss_names = ("Loss", "Heating Rate Loss", "Direct Heating Rate Loss", "Flux Loss")

    for t in range(2650,5000,50):

        checkpoint = torch.load(filename_full_model + str(t))
        print(f"Loaded Model: epoch = {t}")
        model.load_state_dict(checkpoint['model_state_dict'])

        loss = test_loop (test_dataloader, model, loss_functions, loss_names, weight_profile, device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_full_dataloader():

    print("Pytorch version:", torch.__version__)
    device = "cpu"
    print(f"Using {device} device")

    datadir     = "/data-T1/hws/tmp/"
    batch_size = 2048 
    n_channel = 30
    n_constituent = 8

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

    filename_full_model = datadir + f"/Torch.Dataloader.v2_4." # corresponds to Scattering_v3, two inputs to mass_extinction(t,p)

    #filename_full_model = datadir + f"/Torch.Dataloader.v2_5." # corresponds to Scattering_v3, two inputs to mass_extinction(t,p)

    years = ("2009", "2015", "2020")

    for year in years:
        test_input_dir = f"/data-T1/hws/CAMS/processed_data/testing/{year}/"
        months = [str(m).zfill(2) for m in range(1,13)]
        test_input_files = [f'{test_input_dir}Flux_sw-{year}-{month}.nc' for month in months]
        #test_input_files = ["/data-T1/hws/tmp/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.2.nc"]


        weight_profile = get_weight_profile_2(device)
        test_dataset = RT_data_hws_2.RTDataSet(test_input_files,n_channel)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, 
                                                    shuffle=False,
                                                            num_workers=1)


        loss_functions = (loss_henry_full_wrapper, loss_heating_rate_full_wrapper,
                        loss_heating_rate_direct_full_wrapper, loss_flux_full_wrapper)
        loss_names = ("Loss", "Heating Rate Loss", "Direct Heating Rate Loss", "Flux Loss")

        print(f"Testing error, Year = {year}")
        for t in range(180,460,5):

            checkpoint = torch.load(filename_full_model + str(t), map_location=torch.device(device))
            print(f"Loaded Model: epoch = {t}")
            model.load_state_dict(checkpoint['model_state_dict'])

            print(f"Total number of parameters = {n_parameters}", flush=True)
            #print(f"Spectral decomposition weights = {model.spectral_net.weight}", flush=True)

            loss = test_loop (test_dataloader, model, loss_functions, loss_names, weight_profile, device)
    

if __name__ == "__main__":
    #train_direct_only()
    #train_full()
    #
    #test_full()

    
    #train_full_dataloader()
    test_full_dataloader()