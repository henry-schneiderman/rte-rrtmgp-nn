import numpy as np
import torch
from torch import nn
import torch.autograd.profiler as profiler

from RT_data_hws import load_data_direct_pytorch, load_data_full_pytorch_2, absorbed_flux_to_heating_rate

eps_1 = 0.0000001

class MLP(nn.Module):
    """
    Multi Layer Perceptron (MLP) module

    Fully connected layers
    
    Uses ReLU activation for hidden units
    No activation for output unit
    
    Initialization of all weights with uniform distribution with 'lower' and 'upper'
    bounds. Defaults to -0.1 < x < 0.1
    
    Hidden units initial bias with uniform distribution 0.9 < x < 1.1
    Output unit initial bias with uniform distribution -0.1 < x <0.1
    """

    def __init__(self, n_hidden, n_input, n_output, device, lower=-0.1, upper=0.1):
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
        self.relu = nn.ReLU()
        self.output = nn.Linear(n_last, n_output, bias=True, device=device)
        torch.nn.init.uniform_(self.output.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.output.bias, a=-0.1, b=0.1)

    def forward(self, x):
        for hidden in self.hidden:
            x = hidden(x)
            x = self.relu(x)
        return self.output(x)

class LayerDistributed(nn.Module):
    """
    Applies a nn.Module to an array of atmospheric layers

    Adapted from https://stackoverflow.com/questions/62912239/tensorflows-timedistributed-equivalent-in-pytorch

    The input and output may be a list of tensors
    Each tensor has dimensions: (n_samples, n_layers, input / output dimensions. . .)
    """

    def __init__(self, module):
        super(LayerDistributed, self).__init__()
        self.module = module

    def forward(self, x):

        if torch.is_tensor(x):
            shape = x.shape
            n_sample = shape[0]
            n_layer = shape[1]

            squashed_input = x.contiguous().view(n_sample*n_layer, *shape[2:]) 
        else: 
            # Input is a list of tensors. Squash each individually
            squashed_input = []
            for xx in x:
                # Squash samples and layers into a single axis
                shape = xx.shape
                n_sample = shape[0]
                n_layer = shape[1]

                xx_reshape = xx.contiguous().view(n_sample*n_layer, *shape[2:])
                squashed_input.append(xx_reshape)

        y = self.module(squashed_input)

        # Reshape y

        if torch.is_tensor(y):
            shape = y.shape
            unsquashed_output = y.contiguous().view(n_sample, n_layer, *shape[1:]) 
        else:
            # Output is a list of tensors. Unsquash each individually
            unsquashed_output = []
            for yy in y:
                shape = yy.shape
                yy_reshaped = yy.contiguous().view(n_sample, n_layer, *shape[1:])
                unsquashed_output.append(yy_reshaped)

        return unsquashed_output
    

class Extinction(nn.Module):
    """ 
    For a given layer, computes optical depth for each atmospheric constituent for each channel

    Inputs:
        Mass of each atmospheric constituent in layer
        Temperature, pressure, and log(pressure)

    Outputs
        Optical depth of each constituent in layer
    """
    def __init__(self, n_channel, device):
        super(Extinction, self).__init__()
        self.n_channel = n_channel
        self.device = device

        # Computes extinction coeffient for each constituent for each channel
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

        # exp() activation forces coefficient to always be positive
        # and never negative or zero
        self.exp = torch.exp

        # Modifies each extinction coeffient as a function of temperature, pressure
        # and ln(pressure)
        # Seeks to model pressuring broadening of atmospheric absorption lines
        # Single network for each constituent
        self.net_ke_h2o = MLP(n_hidden=(6,4,4),n_input=3,n_output=1,device=device)
        self.net_ke_o3  = MLP(n_hidden=(6,4,4),n_input=3,n_output=1,device=device)
        self.net_ke_co2 = MLP(n_hidden=(6,4,4),n_input=3,n_output=1,device=device)
        self.net_ke_u   = MLP(n_hidden=(6,4,4),n_input=3,n_output=1,device=device)
        self.net_ke_n2o = MLP(n_hidden=(6,4,4),n_input=3,n_output=1,device=device)
        self.net_ke_ch4 = MLP(n_hidden=(6,4,4),n_input=3,n_output=1,device=device)

        self.sigmoid = nn.Sigmoid()

        # Filters select which channels each constituent contributes to
        # Follows similiar assignment of bands as
        # Table A2 in Pincus, R., Mlawer, E. J., &
        # Delamere, J. S. (2019). Balancing accuracy, efficiency, and flexibility in
        # radiation calculations for dynamical models. Journal of Advances in Modeling
        # Earth Systems, 11,3074–3089. https://doi.org/10.1029/2019MS001621

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

    def forward(self, x):
        temperature_pressure_log_pressure, constituents = x

        c = constituents
        t_p = temperature_pressure_log_pressure

        a = self.exp
        b = self.sigmoid

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
    

class DirectTransmission(nn.Module):
    """ Only Computes Direct Transmission Coefficient """
    def __init__(self):
        super(DirectTransmission, self).__init__()

    def forward(self, x):
        mu_direct, tau = x
        
        tau_total = torch.sum(tau,dim=2,keepdim=False)
        t_direct = torch.exp(-tau_total / (mu_direct + eps_1))
        return t_direct


class Scattering(nn.Module):
    """ 
    For a given atmospheric layer, computes radiative coefficients (transmission, reflection,
    and absorption) for both direct and diffuse inputs for each channel.
    Uses a separate MLP for each channel.

    Inputs:

        Cosine of zenith angle: mu_direct, mu_diffuse

        Optical depth of each constituent for each channel: tau

        Mass of each constituent: constituents


    Outputs:

        Direct transmission coefficients of 
            the layer: t_direct, t_diffuse 

        The layer's split of the extinguised  
            radiation into transmitted, reflected,
            and absorbed components. The transmitted
            and reflected components are diffuse. Components 
            sum to 1.0: e_split_direct, e_split_diffuse 

    Note: extinguished radiation is that radiation which is not directly transmitted

    """
    def __init__(self, n_channel, n_constituent, device):
        super(Scattering, self).__init__()
        self.n_channel = n_channel

        """ Models split of extinguished radiation into absorbed, transmitted, and
        reflected components """
        n_hidden = [5, 4, 4]
        # For direct input
        # Has additional input for zenith angle ('mu_direct')
        self.net_direct = nn.ModuleList([MLP(n_hidden, n_constituent + 1,3,device,lower=-1.0,upper=1.0) for _ in range(self.n_channel)])
        # For diffuse input
        self.net_diffuse = nn.ModuleList([MLP(n_hidden, n_constituent,3,device, lower=-1.0,upper=1.0) for _ in range(self.n_channel)])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        tau, mu_direct, mu_diffuse, constituents = x


        tau_total = torch.sum(tau, dim=2, keepdims=False)

        t_direct = torch.exp(-tau_total / (mu_direct + eps_1))
        t_diffuse = torch.exp(-tau_total / (mu_diffuse + eps_1))

        constituents_direct = constituents / (mu_direct + eps_1)
        constituents_diffuse = constituents 
        constituents_direct = torch.concat((constituents_direct, mu_direct),dim=1)

        e_split_direct = [self.softmax(net(constituents_direct)) for net in self.net_direct]
        e_split_diffuse = [self.softmax(net(constituents_diffuse)) for net in self.net_diffuse]

        e_split_direct = torch.stack(e_split_direct, dim=1)
        e_split_diffuse = torch.stack(e_split_diffuse, dim=1)

        layer_properties = [t_direct, t_diffuse, e_split_direct, e_split_diffuse]

        return layer_properties
    
def adding_doubling (t_direct, t_diffuse, e_split_direct, e_split_diffuse, r_bottom_direct, r_bottom_diffuse, a_bottom_direct, a_bottom_diffuse):
    """
    Multireflection between a single layer and a surface.

    Naming conventions:
     
    The prefixes -- t, e, r, a -- correspond respectively to transmission,
    extinction, reflection, absorption.

    The suffixes "_direct" and "_diffuse" specify the type of input radiation. 
    Note, that direct radiation may be transformed into diffuse radiation,
    through reflection or multi-reflection
    
    Input and Output Shape:
        Tensor with shape (n_samples, n_channels, . . .)

    Arguments:

        t_direct, t_diffuse - Direct transmission coefficients of 
            the layer. 

        e_split_direct, e_split_diffuse - The layer's split of extinguised  
            radiation into transmitted, reflected,
            and absorbed components. These components 
            sum to 1.0. Also, transmitted and reflected components are always
            diffuse.
            
        r_bottom_direct, r_bottom_diffuse - The reflection 
            coefficients of the surface.

        a_bottom_direct, a_bottom_diffuse - The absorption coefficients
            of the surface. 
            
    Returns:

        t_multi_direct, t_multi_diffuse - The layer's transmission coefficients of 
            radiation that is multi-reflected (as opposed to directly transmitted, 
            e.g., t_direct, t_diffuse)

        r_multi_direct, r_multi_diffuse - The layer's reflection coefficients 
            after accounting for multi-reflection with the surface

        r_bottom_multi_direct, r_bottom_multi_diffuse - The reflection coefficients for
            the surface after accounting for multi-reflection with the layer

        a_top_multi_direct, a_top_multi_diffuse - The absorption coefficients of 
            the layer after multi-reflection with surface

        a_bottom_multi_direct, a_bottom_multi_diffuse - The absorption coefficients 
            of the surface after multi-reflection with the layer

    Notes:

        When merging the multireflected layer and the surface into a new "surface":
        r = r_multi (the reflection of the combination is the reflection of layer)
        a = a_top_multi + a_bottom_multi (however, the absorption of the combination is the sum)

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

    # Multi-reflection between the layer and surface resolves 
    # a direct input into:
    #   r_multi_direct - total effective reflection at the layer
    #   a_top_multi_direct - absorption at the layer
    #   a_bottom_multi_direct - absorption at the surface

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

class MultiReflection(nn.Module):
    """
    Updates the representation of the atmosphere by incorporating 
    interactions between layers (multireflection) using the 
    adding-doubling method. 

    The basic idea is that we multi reflect between a single 
    layer and a surface computing their revised radiative coefficients 
    (transmission, reflection, and absorption).
    Then the surface and the layer are merged into a new
    "surface" and the process is repeated with 
    the next layer above. We construct this representation by 
    starting at the original surface and moving upwards layer by layer
    to the top of the atmosphere (toa) 

    Computations are independent across channel.
    """

    def __init__(self):
        super(MultiReflection, self).__init__()

    def forward(self, x):

        radiative_layers, x_surface = x

        t_direct, t_diffuse, e_split_direct, e_split_diffuse = radiative_layers

        r_bottom_direct, r_bottom_diffuse, a_bottom_direct, a_bottom_diffuse = x_surface[:,:,0], x_surface[:,:,1], x_surface[:,:,2], x_surface[:,:,3]

        t_multi_direct = []
        t_multi_diffuse = []
        r_bottom_multi_direct = []
        r_bottom_multi_diffuse = []
        a_top_multi_direct = []
        a_top_multi_diffuse = []

        # Compute starting at the original surface and the first layer and progress upwards
        for l in reversed(range(t_direct.shape[1])):
            multireflected_info = adding_doubling (t_direct[:,l,:], t_diffuse[:,l,:], e_split_direct[:,l,:,:], e_split_diffuse[:,l,:,:], r_bottom_direct, r_bottom_diffuse, a_bottom_direct, a_bottom_diffuse)

            t_multi_direct, t_multi_diffuse, \
            r_multi_direct, r_multi_diffuse, \
            r_bottom_multi_direct, r_bottom_multi_diffuse, \
            a_top_multi_direct, a_top_multi_diffuse, \
            a_bottom_multi_direct, a_bottom_multi_diffuse = multireflected_info

            # Merge the layer and surface forming a new "surface"
            r_bottom_direct = r_multi_direct
            r_bottom_diffuse = r_multi_diffuse
            a_bottom_direct = a_top_multi_direct + a_bottom_multi_direct
            a_bottom_diffuse = a_top_multi_diffuse + a_bottom_multi_diffuse

            t_multi_direct.append(t_multi_direct)
            t_multi_diffuse.append(t_multi_diffuse)
            r_bottom_multi_direct.append(r_bottom_multi_direct)
            r_bottom_multi_diffuse.append(r_bottom_multi_diffuse)
            a_top_multi_direct.append(a_top_multi_direct)
            a_top_multi_diffuse.append(a_top_multi_diffuse)

        t_multi_direct= torch.stack(t_multi_direct, dim=1)
        t_multi_diffuse = torch.stack(t_multi_diffuse, dim=1)
        r_bottom_multi_direct = torch.stack(r_bottom_multi_direct, dim=1)
        r_bottom_multi_diffuse = torch.stack(r_bottom_multi_diffuse, dim=1)
        a_top_multi_direct = torch.stack(a_top_multi_direct, dim=1)
        a_top_multi_diffuse = torch.stack(a_top_multi_diffuse, dim=1)

        t_multi_direct = torch.flip(t_multi_direct, dims=(1,))
        t_multi_diffuse = torch.flip(t_multi_diffuse, dims=(1,))
        r_bottom_multi_direct = torch.flip(r_bottom_multi_direct, dims=(1,))
        r_bottom_multi_diffuse = torch.flip(r_bottom_multi_diffuse, dims=(1,))
        a_top_multi_direct = torch.flip(a_top_multi_direct, dims=(1,))
        a_top_multi_diffuse = torch.flip(a_top_multi_diffuse, dims=(1,))

        multireflected_layers = [t_direct, t_diffuse, t_multi_direct, t_multi_diffuse, r_bottom_multi_direct,r_bottom_multi_diffuse, a_top_multi_direct, a_top_multi_diffuse]

        return [multireflected_layers, r_multi_direct]
    

class Propagation(nn.Module):
    """
    Propagate flux from the top of the atmosphere to the
    surface.
    The multireflection layer parameters allows this to be
    done in one pass
    """
    def __init__(self,n_channel):
        super().__init__()
        super(Propagation, self).__init__()
        self.n_channel = n_channel

    def forward(self, x):

        multireflected_layers, r_multi_direct, input_flux = x

        t_direct, t_diffuse, \
        t_multi_direct, t_multi_diffuse, \
        r_bottom_multi_direct, r_bottom_multi_diffuse, \
        a_top_multi_direct, a_top_multi_diffuse  = multireflected_layers

        # Assign all 3 fluxes above the top layer
        input_flux_direct, input_flux_diffuse = input_flux
        flux_down_direct = [input_flux_direct]
        flux_down_diffuse = [input_flux_diffuse]
        flux_up_diffuse = [input_flux_direct * r_multi_direct]

        # Will assign flux absorbed at each layer
        flux_absorbed = []

        for l in range(t_direct.shape[1]):

            flux_absorbed.append(input_flux_direct * a_top_multi_direct[:,l] + 
                        input_flux_diffuse * a_top_multi_diffuse[:,l])

            # Will want this later when incorporate surface interactions
            #flux_absorbed_bottom = input_flux_direct * a_bottom_multi_direct + \
            #input_flux_diffuse * a_bottom_multi_diffuse

            flux_down_direct.append(input_flux_direct * t_direct[:,l])
            flux_down_diffuse.append(input_flux_direct * t_multi_direct[:,l] + 
                                    input_flux_diffuse * (t_diffuse[:,l] + t_multi_diffuse[:,l]))
            flux_up_diffuse.append(input_flux_direct * 
                                         r_bottom_multi_direct[:,l] 
                                         + input_flux_diffuse * 
                                         r_bottom_multi_diffuse[:,l])
            
            input_flux_direct = flux_down_direct[-1]
            input_flux_diffuse = flux_down_diffuse[-1]
        

        flux_down_direct = torch.stack(flux_down_direct,dim=1)
        flux_down_diffuse = torch.stack(flux_down_diffuse,dim=1)
        flux_up_diffuse = torch.stack(flux_up_diffuse,dim=1)
        flux_absorbed = torch.stack(flux_absorbed,dim=1)

        flux_down_direct = torch.sum(flux_down_direct,dim=2,keepdim=False)
        flux_down_diffuse = torch.sum(flux_down_diffuse,dim=2,keepdim=False)      
        flux_up_diffuse = torch.sum(flux_up_diffuse,dim=2,keepdim=False)  
        flux_absorbed = torch.sum(flux_absorbed,dim=2,keepdim=False)  

        return [flux_down_direct, flux_down_diffuse, flux_up_diffuse, flux_absorbed]


class DownwardPropagationDirect(nn.Module):
    def __init__(self,n_channel):
        super(DownwardPropagationDirect, self).__init__()
        self.n_channel = n_channel

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
        self.extinction_net = LayerDistributed(Extinction(n_channel,device))
        self.layer_properties_net = LayerDistributed(DirectTransmission())
        self.downward_propagate = DownwardPropagationDirect(n_channel)

    def forward(self, x):
        mu_direct, temperature_pressure_log_pressure, constituents = x[:,:,0:1], x[:,:,1:4], x[:,:,4:]
        #with profiler.record_function("Spectral Decomposition"):
        one = torch.unsqueeze(torch.ones((mu_direct.shape[0]),dtype=torch.float32,device=self.device), 1)
        flux_down_above_direct_channels = self.softmax(self.spectral_net(one))
        #with profiler.record_function("Optical Depth"):
        tau = self.extinction_net((temperature_pressure_log_pressure, constituents))
        #with profiler.record_function("Layer Properties"):
        t_direct = self.layer_properties_net((mu_direct, tau))
        #with profiler.record_function("Downward Propagate"):
        flux_down_direct_channels = self.downward_propagate((flux_down_above_direct_channels,t_direct))
        flux_down_direct = torch.sum(flux_down_direct_channels,dim=2,keepdim=False)
        return flux_down_direct
    


class FullNet(nn.Module):

    def __init__(self, n_channel, n_constituent, device):
        super(FullNet, self).__init__()
        self.device = device
        self.n_channel = n_channel

        self.mu_diffuse_net = nn.Linear(1,1,bias=False,device=device)
        torch.nn.init.uniform_(self.mu_diffuse_net.weight, a=0.4, b=0.6)
        self.sigmoid = nn.Sigmoid()

        self.spectral_net = nn.Linear(1,n_channel,bias=False,device=device)
        torch.nn.init.uniform_(self.spectral_net.weight, a=0.4, b=0.6)
        self.softmax = nn.Softmax(dim=-1)

        self.extinction_net = LayerDistributed(Extinction(n_channel,device))

        self.scattering_net = LayerDistributed(Scattering(n_channel,n_constituent,device))

        self.multireflection_net = MultiReflection()

        self.propagation_net = Propagation(n_channel)

    def forward(self, x):
        x_layers, x_surface = x
        mu_direct, temperature_pressure_log_pressure, constituents = x_layers[:,:,0:1], x_layers[:,:,1:4], x_layers[:,:,4:12]

        # Diffuse Zenith Angle
        #with profiler.record_function("Mu Diffuse"):
        one = torch.ones((mu_direct.shape[0],1),dtype=torch.float32,device=self.device)
        mu_diffuse = self.sigmoid(self.mu_diffuse_net(one))
        mu_diffuse = mu_diffuse.repeat([1,mu_direct.shape[1]])
        mu_diffuse = torch.unsqueeze(mu_diffuse,dim=2)

        # Optical Depth
        #with profiler.record_function("Extinction"):
        tau = self.extinction_net((temperature_pressure_log_pressure, constituents))

        # Transmission, Reflection, and Absorption for each individual layer
        #with profiler.record_function("Scattering"):
        radiative_layers = self.scattering_net((tau, mu_direct, mu_diffuse, constituents))

        # Interaction (multireflection) among all layers
        #with profiler.record_function("Multireflection"):
        multireflected_layers = self.multireflection_net([radiative_layers,x_surface])

        # Decomposition of flux into spectral channels
        #with profiler.record_function("Flux Propagatation"):
        input_flux_direct = self.softmax(self.spectral_net(one))
        input_flux_diffuse = torch.zeros((mu_direct.shape[0], self.n_channel),dtype=torch.float32,device=self.device)
        input_flux = [input_flux_direct, input_flux_diffuse]

        # Propagation of flux through layers along spectral channels
        flux = self.propagation_net(*multireflected_layers, input_flux)

        flux_down_direct, flux_down_diffuse, flux_up_diffuse, flux_absorbed = flux
        flux_down = flux_down_direct + flux_down_diffuse
        flux_up = flux_up_diffuse
        return [flux_down_direct, flux_down, flux_up, flux_absorbed]

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

def loss_henry(y_true, flux_absorbed_true, y_pred, toa_weighting_profile, delta_pressure, weight_profile):
    flux_down_direct_pred, flux_down_pred, flux_up_pred, flux_absorbed_pred = y_pred
    flux_down_direct_true, flux_down_true, flux_up_true = y_true[:,:,0], y_true[:,:,1], y_true[:,:,2]
    flux_pred = torch.concat((flux_down_pred,flux_up_pred),dim=1)
    flux_true = torch.concat((flux_down_true,flux_up_true),dim=1)
    #weight_profile_x2 = torch.concat((weight_profile,weight_profile),dim=0)
    flux_loss = loss_weighted(flux_true, flux_pred, weight_profile)
    hr_loss = loss_heating_rate_full(flux_absorbed_true, flux_absorbed_pred,  toa_weighting_profile, delta_pressure)
    hr_direct_loss = loss_heating_rate_direct(flux_down_direct_true, flux_down_direct_pred, toa_weighting_profile, delta_pressure)
    alpha   = 1.0e-4
    return alpha * (hr_loss + hr_direct_loss) + (1.0 - alpha) * flux_loss


def train_loop(dataloader, model, optimizer, loss_function, weight_profile):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    loss_string = "Training Loss: "
    for batch, data in enumerate(dataloader):
        # Compute prediction and loss
        y_pred = model(data)

        loss = loss_function(data, y_pred, weight_profile)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 20 == 0:
            loss_value = loss.item()
            loss_string += f" {loss_value:.9f}"

    print (loss_string)

def test_loop(dataloader, model, loss_functions, loss_names, weight_profile):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(dataloader)

    loss = np.zeros(len(loss_functions), dtype=np.float32)

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for data in dataloader:
            y_pred = model(data)
            for i, loss_fn in enumerate(loss_functions):
                loss[i] += loss_fn(data, y_pred, weight_profile).item()

    loss /= num_batches

    print(f"Test Error: ")
    for i, value in loss:
        print(f" {loss_names[i]}: {value:.8f}")

    return loss

def convert_to_tensor(device):
    def inner_conversion(np_ndarray):
        t = torch.from_numpy(np_ndarray).float().to(device)
        return t
    return inner_conversion

def train_direct_only():

    print("Pytorch version:", torch.__version__)
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    datadir     = "/home/hws/tmp/"
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

    checkpoint_period = 100
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
        train_direct_loop(train_dataloader, model, optimizer, weight_profile)
        loss = test_direct_loop(validation_dataloader, model, weight_profile)
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

    datadir     = "/home/hws/tmp/"
    filename_training = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"
    filename_validation = datadir + "/RADSCHEME_data_g224_CAMS_2014.2.nc"
    filename_testing = datadir + "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"
    filename_full_model = datadir + "/Full_Torch."

    batch_size = 2048
    n_channel = 30
    n_constituent = 8
    checkpoint_period = 100
    epochs = 4000

    model = FullNet(n_channel,n_constituent,device).to(device=device)
    optimizer = torch.optim.Adam(model.parameters())

    x_layers, x_surface, x_toa, x_delta_pressure, y_flux, y_flux_absorbed = load_data_full_pytorch_2(filename_training, n_channel)

    weight_profile = get_weight_profile(y_flux, device)

    tensorize = convert_to_tensor(device)
    train_dataset = torch.utils.data.TensorDataset(tensorize(x_layers), 
                                                   tensorize(x_surface), 
                                                   tensorize(x_toa), 
                                                   tensorize(x_delta_pressure), 
                                                   tensorize(y_flux), 
                                                   tensorize(y_flux_absorbed))


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

    x_layers, x_surface, x_toa, x_delta_pressure, y_flux, y_flux_absorbed = load_data_full_pytorch_2(filename_validation, n_channel)
    validation_dataset = torch.utils.data.TensorDataset(tensorize(x_layers),
                                                        tensorize(x_surface), 
                                                        tensorize(x_toa), 
                                                        tensorize(x_delta_pressure), 
                                                        tensorize(y_flux), 
                                                        tensorize(y_flux_absorbed))

    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size, shuffle=True)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    loss_functions = (loss_henry, loss_heating_rate_full, loss_heating_rate_direct,loss_flux)
    loss_names = ("Loss", "Heating Rate Loss", "Direct Heating Rate Loss", "Flux Loss")

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
        train_loop(train_dataloader, model, optimizer, loss_henry, weight_profile)

        loss = test_loop(validation_dataloader, model, loss_functions, loss_names,weight_profile)
        end.record()
        torch.cuda.synchronize()
        print(f" Elapsed time in seconds: {start.elapsed_time(end) / 1000.0}\n")

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

if __name__ == "__main__":
    #train_direct_only()
    train_full()