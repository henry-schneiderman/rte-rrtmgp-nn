import sys
import numpy as np
import torch
from torch import nn
import torch.profiler as profiler
from torch.utils.benchmark import Timer
import time


from RT_data_hws import load_data_direct_pytorch, load_data_full_pytorch,absorbed_flux_to_heating_rate

eps_1 = 0.0000001

class MLP(nn.Module):
    """
    Multi Layer Perceptron (MLP) module
    
    Uses Relu activation for hidden units
    No activation for output unit
    
    Initialization of weights using uniform distribution with 'lower' and 'upper'
    bounds
    
    Hidden units initial bias with uniform distribution 0.9 < x < 1.1
    Output unit initial bias with uniform distribution -0.1 < x <0.1
    """
    def __init__(self, n_hidden, n_input, n_output, dropout, device, lower=-0.1, upper=0.1, is_ELU_activation=False):
        super(MLP, self).__init__()
        self.n_hidden = n_hidden
        self.n_outputs = n_output
        n_last = n_input
        self.hidden = nn.ModuleList()

        for n in n_hidden:
            mod = nn.Linear(n_last, n, bias=True,device=device)
            torch.nn.init.uniform_(mod.weight, a=lower, b=upper)
            # Because of Relu activation, don't want any connections to
            # be prematurely pruned away by becoming negative.
            # Therefore start with a significant positive bias
            torch.nn.init.uniform_(mod.bias, a=0.9, b=1.1) #a=-0.1, b=0.1)
            self.hidden.append(mod)
            n_last = n
        if is_ELU_activation:
            self.activation = nn.Softplus()
        else:
            self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(n_last, n_output, bias=True, device=device)
        torch.nn.init.uniform_(self.output.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.output.bias, a=-0.1, b=0.1)

    @torch.compile
    def forward(self, x):
        for hidden in self.hidden:
            x = hidden(x)
            x = self.dropout(x)
            x = self.activation(x)
        return self.output(x)
    
class OpticalDepth(nn.Module):

    def __init__(self, n_channel, dropout, device):
        super(OpticalDepth, self).__init__()
        self.n_channel = n_channel
        self.device = device

        self.net_lw  = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_iw  = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_h2o = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_o3  = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_co2 = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_u   = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_n2o = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_ch4 = nn.Linear(1,self.n_channel,bias=False,device=device)

        """
        lower = 0.2
        upper = 1.8
        """
        # after change to exp() activation function
        lower = -0.9 # exp(-0.9) = .406
        upper = 0.5 # exp(0.5) = 1.64
        torch.nn.init.uniform_(self.net_lw.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_iw.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_h2o.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_o3.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_co2.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_u.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_n2o.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_ch4.weight, a=lower, b=upper)

        self.net_ke_h2o = MLP(n_hidden=(6,4,4),n_input=3,n_output=1,dropout=dropout,device=device)
        self.net_ke_o3  = MLP(n_hidden=(6,4,4),n_input=3,n_output=1,dropout=dropout,device=device)
        self.net_ke_co2 = MLP(n_hidden=(6,4,4),n_input=3,n_output=1,dropout=dropout,device=device)
        self.net_ke_u   = MLP(n_hidden=(6,4,4),n_input=3,n_output=1,dropout=dropout,device=device)
        self.net_ke_n2o = MLP(n_hidden=(6,4,4),n_input=3,n_output=1,dropout=dropout,device=device)
        self.net_ke_ch4 = MLP(n_hidden=(6,4,4),n_input=3,n_output=1,dropout=dropout,device=device)

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
        
        #self.activation_1 = nn.ReLU()
        # exp() activation forces coefficient to always be positive
        # and never negative or zero
        self.activation_1 = torch.exp
        self.activation_2 = nn.Sigmoid()

    @torch.compile
    def forward(self, x):
        t_p, c = x

        a = self.activation_1
        b = self.activation_2

        """
        tau_lw  = a(self.net_lw (c[:,0:1]))
        tau_iw  = a(self.net_iw (c[:,1:2]))
        tau_h2o = a(self.net_h2o(c[:,2:3])) * self.filter_h2o * b(self.net_ke_h2o(t_p))
        tau_o3  = a(self.net_o3 (c[:,3:4])) * self.filter_o3  * b(self.net_ke_o3 (t_p))
        tau_co2 = a(self.net_co2(c[:,4:5])) * self.filter_co2 * b(self.net_ke_co2(t_p))
        tau_u   = a(self.net_u  (c[:,5:6])) * self.filter_u   * b(self.net_ke_u  (t_p))
        tau_n2o = a(self.net_n2o(c[:,6:7])) * self.filter_n2o * b(self.net_ke_n2o(t_p))
        tau_ch4 = a(self.net_ch4(c[:,7:8])) * self.filter_ch4 * b(self.net_ke_ch4(t_p))
        """

        one = torch.ones((c.shape[0],1), dtype=torch.float32,device=self.device)
        tau_lw  = a(self.net_lw (one)) * (c[:,0:1])
        tau_iw  = a(self.net_iw (one)) * (c[:,1:2])
        tau_h2o = a(self.net_h2o(one)) * (c[:,2:3]) * self.filter_h2o * b(self.net_ke_h2o(t_p))
        tau_o3  = a(self.net_o3 (one)) * (c[:,3:4]) * self.filter_o3  * b(self.net_ke_o3 (t_p))
        tau_co2 = a(self.net_co2(one)) * (c[:,4:5]) * self.filter_co2 * b(self.net_ke_co2(t_p))
        tau_u   = a(self.net_u  (one)) * (c[:,5:6]) * self.filter_u   * b(self.net_ke_u  (t_p))
        tau_n2o = a(self.net_n2o(one)) * (c[:,6:7]) * self.filter_n2o * b(self.net_ke_n2o(t_p))
        tau_ch4 = a(self.net_ch4(one)) * (c[:,7:8]) * self.filter_ch4 * b(self.net_ke_ch4(t_p))

        tau_lw  = torch.unsqueeze(tau_lw,2)
        tau_iw  = torch.unsqueeze(tau_iw,2)
        tau_h2o = torch.unsqueeze(tau_h2o,2)
        tau_o3  = torch.unsqueeze(tau_o3,2)
        tau_co2 = torch.unsqueeze(tau_co2,2)
        tau_u   = torch.unsqueeze(tau_u,2)
        tau_n2o = torch.unsqueeze(tau_n2o,2)
        tau_ch4 = torch.unsqueeze(tau_ch4,2)

        tau = torch.cat([tau_lw, tau_iw, tau_h2o, tau_o3, tau_co2, tau_u, tau_n2o, tau_ch4],dim=2)

        return tau
    
class TimeDistributed(nn.Module):
    """
    Adapted from https://stackoverflow.com/questions/62912239/tensorflows-timedistributed-equivalent-in-pytorch
    Allows inputs and outputs to be lists of tensors
    Input and output elements are: (samples, timesteps, input/output dim 1,
    input / output dim 2, . . .)
    """
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    #@torch.compile
    def forward(self, x):

        if torch.is_tensor(x):
            shape = x.shape
            n_sample = shape[0]
            n_layer = shape[1]

            squashed_input = x.contiguous().view(n_sample*n_layer, *shape[2:])  # (samples * timesteps, input_size)
        else: 
            # Is a list of tensors. Squash each individually
            squashed_input = []
            for xx in x:
                # Squash samples and timesteps into a single axis
                shape = xx.shape
                n_sample = shape[0]
                n_layer = shape[1]

                xx_reshape = xx.contiguous().view(n_sample*n_layer, *shape[2:])  # (samples * timesteps, input_size)
                squashed_input.append(xx_reshape)

        y = self.module(squashed_input)

        # We have to reshape y

        if torch.is_tensor(y):
            shape = y.shape
            unsquashed_output = y.contiguous().view(n_sample, n_layer, *shape[1:])  # (samples, timesteps, output_size)
        else:
            # Is a list of tensors. Unsquash each individually
            unsquashed_output = []
            for yy in y:
                shape = yy.shape
                yy_reshaped = yy.contiguous().view(n_sample, n_layer, *shape[1:])  # (samples, timesteps, output_size)
                unsquashed_output.append(yy_reshaped)

        return unsquashed_output
    
class LayerPropertiesDirect(nn.Module):
    """ Only Computes Direct Transmission Coefficient """
    def __init__(self):
        super(LayerPropertiesDirect, self).__init__()

    @torch.compile
    def forward(self, x):
        mu, tau = x
        
        tau_total = torch.sum(tau,dim=2,keepdim=False)
        t_direct = torch.exp(-tau_total / (mu + eps_1))
        return t_direct


class LayerPropertiesFull_1(nn.Module):
    """ Computes full set of layer properties for
    direct and diffuse transmission, reflection,
    and absorption
    Uses a separate net for each channel
    """
    def __init__(self, n_channel, n_constituent, dropout, device):
        super(LayerPropertiesFull_1, self).__init__()
        self.n_channel = n_channel
        n_hidden = [5, 4, 4]
        """ Computes split of extinguished radiation into absorbed, diffuse transmitted, and
        diffuse reflected """
        self.net_direct = nn.ModuleList([MLP(n_hidden, n_constituent + 1,3,dropout, device,lower=-1.0,upper=1.0) for _ in range(self.n_channel)])
        self.net_diffuse = nn.ModuleList([MLP(n_hidden, n_constituent,3,dropout, device, lower=-1.0,upper=1.0) for _ in range(self.n_channel)])
        self.softmax = nn.Softmax(dim=-1)

    @torch.compile
    def forward(self, x):

        tau, mu, mu_bar, constituents = x

        constituents_direct = constituents / (mu + eps_1)
        constituents_direct = torch.concat((constituents_direct, mu),dim=1)

        constituents_diffuse = constituents 

        tau_total = torch.sum(tau, dim=2, keepdims=False)

        t_direct = torch.exp(-tau_total / (mu + eps_1))
        t_diffuse = torch.exp(-tau_total / (mu_bar + eps_1))

        e_split_direct_list = [self.softmax(net(constituents_direct)) for net in self.net_direct]
        e_split_diffuse_list = [self.softmax(net(constituents_diffuse)) for net in self.net_diffuse]

        e_split_direct = torch.stack(e_split_direct_list, dim=1)
        e_split_diffuse = torch.stack(e_split_diffuse_list, dim=1)

        layer_properties = [t_direct, t_diffuse, e_split_direct, e_split_diffuse]

        return layer_properties
    
class LayerPropertiesFull_2(nn.Module):
    """ Computes full set of layer properties for
    direct and diffuse transmission, reflection,
    and absorption
    Uses a single net for scattering for all channels
    """
    def __init__(self, n_channel, n_constituent, n_coarse_code, dropout, device):
        super(LayerPropertiesFull_2, self).__init__()
        self.n_channel = n_channel
        self.n_coarse_code = n_coarse_code
        n_hidden = [10, 6, 6, 6]
        """ Computes split of extinguished radiation into absorbed, diffuse transmitted, and
        diffuse reflected """
        self.net_direct = MLP(n_hidden, n_constituent + 1 + n_coarse_code,
                              3, dropout,
                              device,lower=-1.0,upper=1.0,
                              is_ELU_activation=False)
        self.net_diffuse = MLP(n_hidden, n_constituent + n_coarse_code,
                               3,dropout, device, lower=-1.0,upper=1.0,
                               is_ELU_activation=False) 
        self.softmax = nn.Softmax(dim=-1)

        self.coarse_code_template = torch.ones((1,n_channel,n_coarse_code), dtype=torch.float32,device=device)
        sigma = 0.25
        const_1 = 1.0 / (sigma * np.sqrt(6.28)) # 2 * pi
        for i in range(n_channel):
            ii = i / (n_channel - 1)
            for j in range(n_coarse_code):
                jj = j / (n_coarse_code - 1)
                self.coarse_code_template[:,i,j] = const_1 * np.exp(-0.5 * np.square((ii - jj)/sigma))

    #@torch.compile
    def forward(self, x):

        tau, mu, mu_bar, constituents = x

        tau_total = torch.sum(tau, dim=2, keepdims=False)

        t_direct = torch.exp(-tau_total / (mu + eps_1))
        t_diffuse = torch.exp(-tau_total / (mu_bar + eps_1))

        constituents_direct = constituents / (mu + eps_1)
        constituents_direct = torch.concat((constituents_direct, mu),dim=1)

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

        e_split_direct = self.softmax(self.net_direct(constituents_direct)) 
        e_split_diffuse = self.softmax(self.net_diffuse(constituents_diffuse))

        layer_properties = [t_direct, t_diffuse, e_split_direct, e_split_diffuse]

        return layer_properties
    
@torch.compile
def propagate_layer_up (t_direct, t_diffuse, e_split_direct, e_split_diffuse, r_bottom_direct, r_bottom_diffuse, a_bottom_direct, a_bottom_diffuse):
    """
    Combines the properties of two atmospheric layers within a column: 
    usually a shallow "top layer" and a thicker "bottom layer" where this "bottom layer" spans all the 
    layers beneath the top layer including the surface. Computes the impact of multi-reflection between these layers.

    Naming conventions:
     
    The prefixes -- t, e, r, a -- correspond respectively to transmission,
    extinction, reflection, absorption.

    The suffixes "_direct" and "_diffuse" specify the type of input radiation. 
    Note, that direct radiation may be transformed into diffuse radiation,
    through reflection or multi-reflection
    
    Input and Output Shape:
        Tensor with shape (n_batches, n_channels)

    Arguments:

        t_direct, t_diffuse - Direct transmission coefficients for 
            the top layer. 

        e_split_direct, e_split_diffuse - The split of extinguised  
            radiation into transmitted, reflected,
            and absorbed components. These components sum to 1.0.
            Also, transmitted and reflected components are always
            diffuse.
            
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
    e_t_direct, e_r_direct, e_a_direct = e_split_direct[:,:,0], e_split_direct[:,:,1],e_split_direct[:,:,2]
    e_t_diffuse, e_r_diffuse, e_a_diffuse = e_split_diffuse[:,:,0], e_split_diffuse[:,:,1],e_split_diffuse[:,:,2]

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

    return [t_multi_direct, t_multi_diffuse, 
            r_multi_direct, r_multi_diffuse, 
            r_bottom_multi_direct, r_bottom_multi_diffuse, 
            a_top_multi_direct, a_top_multi_diffuse, 
            a_bottom_multi_direct, a_bottom_multi_diffuse]


class UpwardPropagation(nn.Module):
    def __init__(self):
        super(UpwardPropagation, self).__init__()

    @torch.compile
    def forward(self, x):

        #layer_properties, surface_properties = x
        layer_properties = x[0]
        surface_properties = x[1]

        #t_direct[n,n_layer,n_channel]
        #t_direct, t_diffuse, e_split_direct, e_split_diffuse = layer_properties
        t_direct=layer_properties[0]
        t_diffuse=layer_properties[1]
        e_split_direct=layer_properties[2]
        e_split_diffuse=layer_properties[3]

        #r_bottom_direct, r_bottom_diffuse, a_bottom_direct, a_bottom_diffuse = surface_properties[:,:,0], surface_properties[:,:,1], surface_properties[:,:,2], surface_properties[:,:,3]
        r_bottom_direct= surface_properties[:,:,0]
        r_bottom_diffuse= surface_properties[:,:,1]
        a_bottom_direct= surface_properties[:,:,2]
        a_bottom_diffuse= surface_properties[:,:,3]

        t_multi_direct_list = []
        t_multi_diffuse_list = []
        r_bottom_multi_direct_list = []
        r_bottom_multi_diffuse_list = []
        a_top_multi_direct_list = []
        a_top_multi_diffuse_list = []

        for l in reversed(torch.arange(start=0, end=t_direct.shape[1])):
            multirelection_layer = propagate_layer_up (t_direct[:,l,:], t_diffuse[:,l,:], e_split_direct[:,l,:,:], e_split_diffuse[:,l,:,:], r_bottom_direct, r_bottom_diffuse, a_bottom_direct, a_bottom_diffuse)

            t_multi_direct, t_multi_diffuse, \
            r_multi_direct, r_multi_diffuse, \
            r_bottom_multi_direct, r_bottom_multi_diffuse, \
            a_top_multi_direct, a_top_multi_diffuse, \
            a_bottom_multi_direct, a_bottom_multi_diffuse= multirelection_layer

            r_bottom_direct = r_multi_direct
            r_bottom_diffuse = r_multi_diffuse
            a_bottom_direct = a_top_multi_direct + a_bottom_multi_direct
            a_bottom_diffuse = a_top_multi_diffuse + a_bottom_multi_diffuse

            t_multi_direct_list.append(t_multi_direct)
            t_multi_diffuse_list.append(t_multi_diffuse)
            r_bottom_multi_direct_list.append(r_bottom_multi_direct)
            r_bottom_multi_diffuse_list.append(r_bottom_multi_diffuse)
            a_top_multi_direct_list.append(a_top_multi_direct)
            a_top_multi_diffuse_list.append(a_top_multi_diffuse)

        t_multi_direct= torch.stack(t_multi_direct_list, dim=1)
        t_multi_diffuse = torch.stack(t_multi_diffuse_list, dim=1)
        r_bottom_multi_direct = torch.stack(r_bottom_multi_direct_list, dim=1)
        r_bottom_multi_diffuse = torch.stack(r_bottom_multi_diffuse_list, dim=1)
        a_top_multi_direct = torch.stack(a_top_multi_direct_list, dim=1)
        a_top_multi_diffuse = torch.stack(a_top_multi_diffuse_list, dim=1)

        t_multi_direct = torch.flip(t_multi_direct, dims=(1,))
        t_multi_diffuse = torch.flip(t_multi_diffuse, dims=(1,))
        r_bottom_multi_direct = torch.flip(r_bottom_multi_direct, dims=(1,))
        r_bottom_multi_diffuse = torch.flip(r_bottom_multi_diffuse, dims=(1,))
        a_top_multi_direct = torch.flip(a_top_multi_direct, dims=(1,))
        a_top_multi_diffuse = torch.flip(a_top_multi_diffuse, dims=(1,))

        multireflection_layer_properties = [t_direct, t_diffuse, t_multi_direct, t_multi_diffuse, r_bottom_multi_direct,r_bottom_multi_diffuse, a_top_multi_direct, a_top_multi_diffuse]

        return [multireflection_layer_properties, r_bottom_direct]
    

class DownwardPropagation(nn.Module):
    def __init__(self,n_channel):
        super().__init__()
        super(DownwardPropagation, self).__init__()
        self.n_channel = n_channel

    @torch.compile
    def forward(self, x):

        multireflection_layer_properties, r_multi_direct, input_fluxes = x
        flux_down_above_direct, flux_down_above_diffuse =  input_fluxes

        t_direct, t_diffuse, \
        t_multi_direct, t_multi_diffuse, \
        r_bottom_multi_direct, r_bottom_multi_diffuse, \
        a_top_multi_direct, a_top_multi_diffuse  = multireflection_layer_properties

        absorbed_flux_top = []
        # Assign all 3 fluxes above the top layer
        flux_down_below_direct = [flux_down_above_direct]
        flux_down_below_diffuse = [flux_down_above_diffuse]
        flux_up_below_diffuse = [flux_down_above_direct * r_multi_direct]
        
        for l in range(t_direct.shape[1]):

            absorbed_flux_top.append(flux_down_above_direct * a_top_multi_direct[:,l] + 
                        flux_down_above_diffuse * a_top_multi_diffuse[:,l])

            # Will want this later when incorporate surface interactions
            #absorbed_flux_bottom = flux_down_above_direct * a_bottom_multi_direct + \
            #flux_down_above_diffuse * a_bottom_multi_diffuse

            flux_down_below_direct.append(flux_down_above_direct * t_direct[:,l])
            flux_down_below_diffuse.append(flux_down_above_direct * t_multi_direct[:,l] + 
                                    flux_down_above_diffuse * (t_diffuse[:,l] + t_multi_diffuse[:,l]))
            flux_up_below_diffuse.append(flux_down_above_direct * 
                                         r_bottom_multi_direct[:,l] 
                                         + flux_down_above_diffuse * 
                                         r_bottom_multi_diffuse[:,l])
            
            flux_down_above_direct = flux_down_below_direct[-1]
            flux_down_above_diffuse = flux_down_below_diffuse[-1]
        

        flux_down_direct = torch.stack(flux_down_below_direct,dim=1)
        flux_down_diffuse = torch.stack(flux_down_below_diffuse,dim=1)
        flux_up_diffuse = torch.stack(flux_up_below_diffuse,dim=1)
        absorbed_flux = torch.stack(absorbed_flux_top,dim=1)

        flux_down_direct = torch.sum(flux_down_direct,dim=2,keepdim=False)
        flux_down_diffuse = torch.sum(flux_down_diffuse,dim=2,keepdim=False)      
        flux_up_diffuse = torch.sum(flux_up_diffuse,dim=2,keepdim=False)  
        absorbed_flux = torch.sum(absorbed_flux,dim=2,keepdim=False)  

        return [flux_down_direct, flux_down_diffuse, flux_up_diffuse, absorbed_flux]


class DownwardPropagationDirect(nn.Module):
    def __init__(self,n_channel):
        super(DownwardPropagationDirect, self).__init__()
        self.n_channel = n_channel

    @torch.compile
    def forward(self, x):
        flux_down_above_direct_channels, t_direct = x
        flux_down_direct_channels = [flux_down_above_direct_channels]
        flux = flux_down_above_direct_channels
        for i in range(t_direct.size(1)):
            flux = flux * t_direct[:,i,:]
            flux_down_direct_channels.append(flux)

        output = torch.stack(flux_down_direct_channels,dim=1)

        return output


class DirectDownwardNet(nn.Module):

    def __init__(self, n_channel, device):
        super(DirectDownwardNet, self).__init__()
        self.device = device
        self.spectral_net = nn.Linear(1,n_channel,bias=False,device=device)
        torch.nn.init.uniform_(self.spectral_net.weight, a=0.4, b=0.6)
        self.softmax = nn.Softmax(dim=-1)
        self.optical_depth_net = TimeDistributed(OpticalDepth(n_channel,device))
        self.layer_properties_net = TimeDistributed(LayerPropertiesDirect())
        self.downward_propagate = DownwardPropagationDirect(n_channel)

    @torch.compile
    def forward(self, x):
        mu, temperature_pressure, constituents = x[:,:,0:1], x[:,:,1:4], x[:,:,4:]
        #with profiler.record_function("Spectral Decomposition"):
        one = torch.unsqueeze(torch.ones((mu.shape[0]),dtype=torch.float32,device=self.device), 1)
        flux_down_above_direct_channels = self.softmax(self.spectral_net(one))
        #with profiler.record_function("Optical Depth"):
        tau = self.optical_depth_net((temperature_pressure, constituents))
        #with profiler.record_function("Layer Properties"):
        t_direct = self.layer_properties_net((mu, tau))
        #with profiler.record_function("Downward Propagate"):
        flux_down_direct_channels = self.downward_propagate((flux_down_above_direct_channels,t_direct))
        flux_down_direct = torch.sum(flux_down_direct_channels,dim=2,keepdim=False)
        return flux_down_direct
    


class FullNet(nn.Module):

    def __init__(self, n_channel, n_constituent, dropout, device):
        super(FullNet, self).__init__()
        self.device = device
        self.n_channel = n_channel
        n_coarse_code = 3
        self.mu_bar_net = nn.Linear(1,1,bias=False,device=device)
        torch.nn.init.uniform_(self.mu_bar_net.weight, a=0.4, b=0.6)
        self.sigmoid = nn.Sigmoid()
        self.spectral_net = nn.Linear(1,n_channel,bias=False,device=device)
        torch.nn.init.uniform_(self.spectral_net.weight, a=0.4, b=0.6)
        self.softmax = nn.Softmax(dim=-1)
        self.optical_depth_net = TimeDistributed(OpticalDepth(n_channel,dropout,device))
        if False:
            self.layer_properties_net = TimeDistributed(LayerPropertiesFull_1(n_channel,n_constituent, dropout, self.device))
        else:
            self.layer_properties_net = TimeDistributed(LayerPropertiesFull_2(n_channel,n_constituent, n_coarse_code, dropout, self.device))
        #self.upward_net = torch.jit.script(UpwardPropagation())
        self.upward_net = UpwardPropagation()
        self.downward_net = DownwardPropagation(n_channel)

    def forward(self, x):
        x, surface_properties = x
        mu, temperature_pressure, constituents = x[:,:,0:1], x[:,:,1:4], x[:,:,4:]
        #with profiler.record_function("Spectral Decomposition"):
        one = torch.ones((mu.shape[0],1),dtype=torch.float32,device=self.device)
        flux_down_above_direct_channels = self.softmax(self.spectral_net(one))
        flux_down_above_diffuse_channels = torch.zeros((mu.shape[0], self.n_channel),dtype=torch.float32,device=self.device)
        mu_bar = self.sigmoid(self.mu_bar_net(one))
        mu_bar = mu_bar.repeat([1,mu.shape[1]])
        mu_bar = torch.unsqueeze(mu_bar,dim=2)
        #with profiler.record_function("Optical Depth"):
        if True:
            tau = self.optical_depth_net([temperature_pressure, constituents])
        else:   
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as p:
                tau = self.optical_depth_net((temperature_pressure, constituents))
            print(p.key_averages().table(
                sort_by="cpu_time_total", row_limit=25))
        #with profiler.record_function("Layer Properties"):
        layer_properties = self.layer_properties_net([tau, mu, mu_bar, constituents])

        if True:
            multireflection_layer_properties = self.upward_net([layer_properties,surface_properties])
        else:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as p:
                multireflection_layer_properties = self.upward_net([layer_properties,surface_properties])
            print(p.key_averages().table(
                sort_by="cpu_time_total", row_limit=25))
        #with profiler.record_function("Downward Propagate"):
        input_fluxes = [flux_down_above_direct_channels, flux_down_above_diffuse_channels]
        multireflection_layer_properties.append(input_fluxes)
        flux_down_direct, flux_down_diffuse, flux_up_diffuse, absorbed_flux = self.downward_net(multireflection_layer_properties)
        flux_down = flux_down_direct + flux_down_diffuse
        flux_up = flux_up_diffuse
        return [flux_down_direct, flux_down, flux_up, absorbed_flux]

    
def loss_weighted(y, y_pred, weight_profile):
    error = torch.mean(torch.square(weight_profile * (y_pred - y)), dim=(0,1), keepdim=False)
    return error

def loss_heating_rate_direct(y, y_pred, toa, delta_pressure):
    absorbed_true = y[:,:-1] - y[:,1:]
    absorbed_pred = y_pred[:,:-1] - y_pred[:,1:]
    heat_true = absorbed_flux_to_heating_rate(absorbed_true, delta_pressure)
    heat_pred = absorbed_flux_to_heating_rate(absorbed_pred, delta_pressure)
    error = torch.sqrt(torch.mean(torch.square(toa * (heat_true - heat_pred)),dim=(0,1),keepdim=False))
    return error

def loss_heating_rate_full(absorbed_true, absorbed_pred, toa, delta_pressure):
    heat_true = absorbed_flux_to_heating_rate(absorbed_true, delta_pressure)
    heat_pred = absorbed_flux_to_heating_rate(absorbed_pred, delta_pressure)
    error = torch.sqrt(torch.mean(torch.square(toa * (heat_true - heat_pred)),dim=(0,1),keepdim=False))
    return error

def loss_ukkonen_direct(y, y_pred, toa, delta_pressure, weight_profile):
    # y(n_examples, n_layers+1)
    loss_flux = loss_weighted (y, y_pred, weight_profile)
    hr_loss = loss_heating_rate_direct(y, y_pred, toa, delta_pressure)
    alpha   = 1.0e-4
    return alpha * hr_loss + (1.0 - alpha) * loss_flux

def loss_henry(y_true, absorbed_flux_true, y_pred, toa_weighting_profile, delta_pressure, weight_profile):
    flux_down_direct_pred, flux_down_pred, flux_up_pred, absorbed_flux_pred = y_pred
    flux_down_direct_true, flux_down_true, flux_up_true = y_true[:,:,0], y_true[:,:,1], y_true[:,:,2]
    flux_pred = torch.concat((flux_down_pred,flux_up_pred),dim=1)
    flux_true = torch.concat((flux_down_true,flux_up_true),dim=1)
    #weight_profile_x2 = torch.concat((weight_profile,weight_profile),dim=0)
    flux_loss = loss_weighted(flux_true, flux_pred, weight_profile)
    hr_loss = loss_heating_rate_full(absorbed_flux_true, absorbed_flux_pred,  toa_weighting_profile, delta_pressure)
    hr_direct_loss = loss_heating_rate_direct(flux_down_direct_true, flux_down_direct_pred, toa_weighting_profile, delta_pressure)
    alpha   = 1.0e-4
    return alpha * (hr_loss + hr_direct_loss) + (1.0 - alpha) * flux_loss

def train_direct_loop(dataloader, model, optimizer, weight_profile):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    loss_string = "Training Loss: "
    for batch, (x, y, toa, delta_pressure) in enumerate(dataloader):
        # Compute prediction and loss
        y_pred = model(x)
        loss = loss_ukkonen_direct(y, y_pred, toa, delta_pressure, weight_profile)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 20 == 0:
            loss_value = loss.item()
            loss_string += f" {loss_value:.9f}"

    print (loss_string)

def train_full_loop(dataloader, model, optimizer, weight_profile):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    loss_string = "Training Loss: "
    for batch, (x, surface_properties, y, absorbed_flux, toa, delta_pressure) in enumerate(dataloader):
        # Compute prediction and loss
        y_pred = model((x,surface_properties))

        loss = loss_henry(y, absorbed_flux, y_pred, toa, delta_pressure, weight_profile)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 20 == 0:
            loss_value = loss.item()
            loss_string += f" {loss_value:.9f}"

    print (loss_string)

def test_direct_loop(dataloader, model, weight_profile):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(dataloader)
    loss = 0.0
    loss_heating_rate = 0.0
    loss_flux = 0.0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for x, y, toa, delta_pressure in dataloader:
            y_pred = model(x)
            loss += loss_ukkonen_direct(y, y_pred, toa, delta_pressure, weight_profile).item()
            loss_heating_rate += loss_heating_rate_direct(y, y_pred, toa, delta_pressure).item()
            loss_flux += loss_weighted(y, y_pred, toa).item()

    loss /= num_batches
    loss_heating_rate /= num_batches
    loss_flux /= num_batches

    print(f"Test Error: \n Loss: {loss:.10f}\n Heating Rate Loss: {loss_heating_rate:.8f}")
    print(f" Flux Loss: {loss_flux:.8f}\n")

    return loss

def test_full_loop(dataloader, model, weight_profile):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(dataloader)
    loss = 0.0
    loss_heating_rate = 0.0
    loss_direct_heating_rate = 0.0
    loss_flux = 0.0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for x, surface_properties, y, absorbed_flux, toa, delta_pressure in dataloader:
            y_pred = model((x,surface_properties))
            loss += loss_henry(y, absorbed_flux, y_pred, toa, delta_pressure, weight_profile).item()

            flux_down_direct_pred, flux_down_pred, flux_up_pred, absorbed_flux_pred = y_pred
            flux_down_direct_true, flux_down_true, flux_up_true = y[:,:,0], y[:,:,1], y[:,:,2]

            loss_direct_heating_rate += loss_heating_rate_direct(flux_down_direct_true, flux_down_direct_pred, toa, delta_pressure).item()
            flux_pred = torch.concat((flux_down_pred,flux_up_pred),dim=1)
            flux_true = torch.concat((flux_down_true,flux_up_true),dim=1)
            loss_flux += loss_weighted(flux_true, flux_pred, toa).item()
            loss_heating_rate += loss_heating_rate_full(absorbed_flux, absorbed_flux_pred, toa, delta_pressure).item()

    loss /= num_batches
    loss_direct_heating_rate /= num_batches
    loss_heating_rate /= num_batches
    loss_flux /= num_batches

    print(f"Test Error: \n Loss: {loss:.10f}\n Heating Rate Loss: {loss_heating_rate:.8f}")
    print(f" Direct Heating Rate Loss: {loss_direct_heating_rate:.8f}")
    print(f" Flux Loss: {loss_flux:.8f}\n")

    return loss
 
def train_direct_only():

    print("Pytorch version:", torch.__version__)
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    datadir     = "/data-T1/hws/tmp/"
    filename_training = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"
    filename_validation = datadir + "/RADSCHEME_data_g224_CAMS_2014.2.nc"
    filename_testing = datadir + "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"
    filename_direct_model = datadir + "/Direct_Torch."

    batch_size = 2048
    n_channel = 30
    n_constituent = 8
    model = DirectDownwardNet(n_channel,device)
    model = model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters())

    checkpoint_period = 50
    epochs = 4000

    x_train, y_train, toa_train, delta_pressure_train = load_data_direct_pytorch(filename_training, n_channel)
    weight_profile = 1.0 / torch.mean(torch.from_numpy(y_train).float().to(device), dim=0, keepdim=True)
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float().to(device), 
                                                   torch.from_numpy(y_train).float().to(device),
                                                   torch.from_numpy(toa_train).float().to(device),
                                                   torch.from_numpy(delta_pressure_train).float().to(device))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

    x_valid, y_valid, toa_valid, delta_pressure_valid = load_data_direct_pytorch(filename_validation, n_channel)
    validation_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_valid).float().to(device), 
                                                        torch.from_numpy(y_valid).float().to(device),
                                                        torch.from_numpy(toa_valid).float().to(device),
                                                        torch.from_numpy(delta_pressure_valid).float().to(device))

    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size, shuffle=True)
    #start = torch.cuda.Event(enable_timing=True)
    #end = torch.cuda.Event(enable_timing=True)

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
        t1 = time.perf_counter()
        #with profiler.profile(with_stack=True, profile_memory=True) as prof:
        #start.record()
        train_direct_loop(train_dataloader, model, optimizer, weight_profile)
        loss = test_direct_loop(validation_dataloader, model, weight_profile)
        t2 = time.perf_counter()
        #end.record()
        #torch.cuda.synchronize()
        #print(f" Elapsed time in seconds: {start.elapsed_time(end) / 1000.0}\n")
        print(f"Elapsed time in seconds = {t2 - t1:.4f}")

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

def get_weight_profile(y,device):
    rsd = torch.from_numpy(y[:,:,1]).float().to(device)
    rsu = torch.from_numpy(y[:,:,2]).float().to(device)
    flux = torch.concat((rsd,rsu),dim=1)
    weight_profile = 1.0 / torch.mean(flux,dim=0,keepdim=True)
    return weight_profile

def train_full():

    print("Pytorch version:", torch.__version__)
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        if device_cap in ((7, 0), (8, 0), (9, 0)):
            gpu_ok = True
            print("GPU-OK")
        else:
            print("GPU SLOW!")

    datadir     = "/data-T1/hws/tmp/"
    filename_training = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"
    filename_validation = datadir + "/RADSCHEME_data_g224_CAMS_2014.2.nc"
    filename_testing = datadir + "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"
    filename_full_model = datadir + "/Full_Torch.2."

    batch_size = 2048
    n_channel = 30
    n_constituent = 8
    #dropout = 0.15 #1000 -1100
    #dropout = 0.075    #1100 - 1150
    dropout = 0.0

    #model = torch.compile(FullNet(n_channel,n_constituent,dropout, device))
    model = FullNet(n_channel,n_constituent,dropout,device)
    model = model.to(device=device)


    optimizer = torch.optim.Adam(model.parameters())

    checkpoint_period = 50
    epochs = 4000

    x_train, surface_properties_train, y_train, absorbed_flux_train, toa_train, delta_pressure_train = load_data_full_pytorch(filename_training, n_channel)
    weight_profile = get_weight_profile(y_train, device)
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float().to(device), 
                                                   torch.from_numpy(surface_properties_train).float().to(device),
                                                   torch.from_numpy(y_train).float().to(device),
                                                torch.from_numpy(absorbed_flux_train).float().to(device),
                                                   torch.from_numpy(toa_train).float().to(device),
                                                   torch.from_numpy(delta_pressure_train).float().to(device))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

    x_valid, surface_properties_valid, y_valid, absorbed_flux_valid, toa_valid, delta_pressure_valid = load_data_full_pytorch(filename_validation, n_channel)
    validation_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_valid).float().to(device), 
                                                        torch.from_numpy(surface_properties_valid).float().to(device),
                                                        torch.from_numpy(y_valid).float().to(device),
                                                        torch.from_numpy(absorbed_flux_valid).float().to(device),
                                                        torch.from_numpy(toa_valid).float().to(device),
                                                        torch.from_numpy(delta_pressure_valid).float().to(device))

    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size, shuffle=True)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    if False:
        t = 0
    else:   
        t = 1150
        checkpoint = torch.load(filename_full_model + str(t))
        print(f"Loaded Model: epoch = {t}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #epoch = checkpoint['epoch']
    while t < epochs:
        t += 1
        print(f"Epoch {t}\n-------------------------------")

        if False:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    #torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as p:
                start.record()
                train_full_loop(train_dataloader, model, optimizer, weight_profile)
                loss = test_full_loop(validation_dataloader, model, weight_profile)
                end.record()
                torch.cuda.synchronize()
            print(p.key_averages().table(
                    sort_by="self_cpu_time_total", row_limit=30))
            print(f" Elapsed time in seconds: {start.elapsed_time(end) / 1000.0}\n")
            sys.stdout.flush()
        else:
            #start.record()
            t1 = time.perf_counter()
            train_full_loop(train_dataloader, model, optimizer, weight_profile)
            loss = test_full_loop(validation_dataloader, model, weight_profile)
            #end.record()
            #torch.cuda.synchronize()
            #print(f" Elapsed time in seconds: {start.elapsed_time(end) / 1000.0}\n")
            t2 = time.perf_counter()
            print(f"Elapsed time in seconds = {t2 - t1:.4f}")
            sys.stdout.flush()
        if t % checkpoint_period == 0:
            torch.save({
            'epoch': t,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, filename_full_model + str(t))
            print(f' Wrote Model: epoch = {t}')

    print("Done!")
    #print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

def test_full():

    print("Pytorch version:", torch.__version__)
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    datadir     = "/data-T1/hws/tmp/"
    filename_training = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"
    filename_validation = datadir + "/RADSCHEME_data_g224_CAMS_2014.2.nc"
    filename_testing = datadir + "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"
    filename_full_model = datadir + "/Full_Torch.2."

    batch_size = 2048
    n_channel = 30
    n_constituent = 8

    x_test, surface_properties_test, y_test, absorbed_flux_test, toa_test, delta_pressure_test = load_data_full_pytorch(filename_testing, n_channel)

    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test).float().to(device), 
                                                   torch.from_numpy(surface_properties_test).float().to(device),
                                                   torch.from_numpy(y_test).float().to(device),
                                                torch.from_numpy(absorbed_flux_test).float().to(device),
                                                   torch.from_numpy(toa_test).float().to(device),
                                                   torch.from_numpy(delta_pressure_test).float().to(device))

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True)

    model = FullNet(n_channel,n_constituent,dropout=0.0, device=device)
    model = model.to(device=device)

    for t in range(1000,1750,50):

        checkpoint = torch.load(filename_full_model + str(t))
        print(f"Loaded Model: epoch = {t}")
        model.load_state_dict(checkpoint['model_state_dict'])

        num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters = {num}")

        model.eval()
        num_batches = len(test_dataloader)

        loss_direct_heating_rate = 0.0
        loss_heating_rate = 0.0
        loss_flux = 0.0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for x, surface_properties, y, absorbed_flux, toa, delta_pressure in test_dataloader:
                y_pred = model((x,surface_properties))

                flux_down_direct_pred, flux_down_pred, flux_up_pred, absorbed_flux_pred = y_pred
                flux_down_direct_true, flux_down_true, flux_up_true = y[:,:,0], y[:,:,1], y[:,:,2]

                loss_direct_heating_rate += loss_heating_rate_direct(flux_down_direct_true, flux_down_direct_pred, toa, delta_pressure).item()
                flux_pred = torch.concat((flux_down_pred,flux_up_pred),dim=1)
                flux_true = torch.concat((flux_down_true,flux_up_true),dim=1)
                loss_flux += loss_weighted(flux_true, flux_pred, toa).item()
                loss_heating_rate += loss_heating_rate_full(absorbed_flux, absorbed_flux_pred, toa, delta_pressure).item()

        loss_direct_heating_rate /= num_batches
        loss_heating_rate /= num_batches
        loss_flux /= num_batches

        print(f"Test Error: \n Heating Rate Loss: {loss_heating_rate:.8f}")
        print(f" Direct Heating Rate Loss: {loss_direct_heating_rate:.8f}")
        print(f" Flux Loss: {loss_flux:.8f}\n")


if __name__ == "__main__":
    #train_direct_only()
    #train_full()
    test_full()