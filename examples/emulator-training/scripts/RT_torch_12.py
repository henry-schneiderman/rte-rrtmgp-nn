import numpy as np
from typing import List
import torch
from torch import nn
import torch.profiler as profiler

from RT_data_hws import load_data_direct_pytorch, load_data_full_pytorch_2, absorbed_flux_to_heating_rate

eps_1 = 0.0000001

class MLP(nn.Module):
    """
    Multi Layer Perceptron (MLP) module

    Fully connected layers
    
    Uses ReLU activation for hidden units
    No activation for output unit
    
    Initialization of all weights with uniform distribution with 'lower' 
    and 'upper' bounds. Defaults to -0.1 < weight < 0.1
    
    Hidden units initial bias with uniform distribution 0.9 < x < 1.1
    Output unit initial bias with uniform distribution -0.1 < x <0.1
    """

    def __init__(self, n_input, n_hidden: List[int], n_output, device, 
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
    Applies a nn.Module independently to an array of atmospheric layers

    Same idea as TensorFlow's TimeDistributed Class
    
    Adapted from:
    https://stackoverflow.com/questions/62912239/tensorflows-timedistributed-equivalent-in-pytorch

    The input and output may each be a single single
    tensor or a list of tensors.

    Each tensor has dimensions: (n_samples, n_layers, data's dimensions. . .)
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

        # exp() activation forces extinction coefficient to always be positive
        # and never negative or zero
        self.exp = torch.exp

        # Modifies each extinction coeffient as a function of temperature, 
        # pressure and ln(pressure)
        # Seeks to model pressuring broadening of atmospheric absorption lines
        # Single network for each constituent
        self.net_ke_h2o = MLP(n_input=3,n_hidden=(6,4,4),n_output=1,
                              device=device)
        self.net_ke_o3  = MLP(n_input=3,n_hidden=(6,4,4),n_output=1,
                              device=device)
        self.net_ke_co2 = MLP(n_input=3,n_hidden=(6,4,4),n_output=1,
                              device=device)
        self.net_ke_u   = MLP(n_input=3,n_hidden=(6,4,4),n_output=1,
                              device=device)
        self.net_ke_n2o = MLP(n_input=3,n_hidden=(6,4,4),n_output=1,
                              device=device)
        self.net_ke_ch4 = MLP(n_input=3,n_hidden=(6,4,4),n_output=1,
                              device=device)

        self.sigmoid = nn.Sigmoid()

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

    def forward(self, x):
        temperature_pressure_log_pressure, constituents = x

        c = constituents
        t_p = temperature_pressure_log_pressure

        a = self.exp
        b = self.sigmoid

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


class Scattering(nn.Module):
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

    def __init__(self, n_channel, n_constituent, device):

        super(Scattering, self).__init__()
        self.n_channel = n_channel

        n_hidden = [5, 4, 4]
        # For direct input
        # Has additional input for zenith angle ('mu_direct')
        self.net_direct = nn.ModuleList(
            [MLP(n_input=n_constituent + 1,
                 n_hidden=n_hidden,
                 n_output=3,
                 device=device,
                 lower=-1.0,upper=1.0) 
             for _ in range(self.n_channel)])
        # For diffuse input
        self.net_diffuse = nn.ModuleList(
            [MLP(n_input=n_constituent, 
                 n_hidden=n_hidden, 
                 n_output=3,
                 device=device, 
                 lower=-1.0,upper=1.0) 
             for _ in range(self.n_channel)])
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        tau, mu_direct, mu_diffuse, constituents = x

        tau_total = torch.sum(tau, dim=2, keepdims=False)

        t_direct = torch.exp(-tau_total / (mu_direct + eps_1))
        t_diffuse = torch.exp(-tau_total / (mu_diffuse + eps_1))

        constituents_direct = constituents / (mu_direct + eps_1)
        constituents_diffuse = constituents 
        constituents_direct = torch.concat((constituents_direct, mu_direct),
                                           dim=1)

        e_split_direct = [self.softmax(net(constituents_direct)) for net 
                          in self.net_direct]
        e_split_diffuse = [self.softmax(net(constituents_diffuse)) for net 
                           in self.net_diffuse]

        e_split_direct = torch.stack(e_split_direct, dim=1)
        e_split_diffuse = torch.stack(e_split_diffuse, dim=1)

        layers = [t_direct, t_diffuse, 
                  e_split_direct, e_split_diffuse]

        return layers
    

class MultiReflection(nn.Module):
    """ 
    Recomputes each layer's radiative coefficients by accounting
    for interaction (multireflection) with all other layers using the 
    Adding-Doubling method (no learning).

    Multi reflects between a surface and a single 
    overhead layer generating their revised radiative coefficients.
    Then merges this surface and layer into a new
    "surface" and we repeat this process with 
    the next layer above and continues
    to the top of the atmosphere (TOA).

    Computations are independent across channel.
     
    The variable prefixes -- t, e, r, a -- correspond respectively to
    transmission, extinction, reflection, and absorption.

    """

    def __init__(self):
        super(MultiReflection, self).__init__()

    def _adding_doubling (t_direct, t_diffuse, 
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
                - t_diffuse is for diffuse input radiation that is directly 
                transmitted.

            e_split_direct, e_split_diffuse - The layer's split of extinguised  
                radiation into transmitted, reflected,
                and absorbed components. These components 
                sum to 1.0. Also, the transmitted and reflected components are 
                always diffuse.
                
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

            See class Propagation for how the multi-reflection
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

        radiative_layers, x_surface = x

        t_direct, t_diffuse, e_split_direct, e_split_diffuse = radiative_layers

        (r_surface_direct, r_surface_diffuse, 
         a_surface_direct, a_surface_diffuse) = (x_surface[:,:,0], 
                                               x_surface[:,:,1], 
                                               x_surface[:,:,2], 
                                               x_surface[:,:,3])

        t_multi_direct = []
        t_multi_diffuse = []
        r_surface_multi_direct = []
        r_surface_multi_diffuse = []
        a_layer_multi_direct = []
        a_layer_multi_diffuse = []

        # Start at the original surface and the first layer and move up
        # one layer for each iteration
        for i in reversed(range(t_direct.shape[1])):
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

            t_multi_direct.append(t_multi_direct)
            t_multi_diffuse.append(t_multi_diffuse)
            r_surface_multi_direct.append(r_surface_multi_direct)
            r_surface_multi_diffuse.append(r_surface_multi_diffuse)
            a_layer_multi_direct.append(a_layer_multi_direct)
            a_layer_multi_diffuse.append(a_layer_multi_diffuse)

        # Stack output in layers
        t_multi_direct= torch.stack(t_multi_direct, dim=1)
        t_multi_diffuse = torch.stack(t_multi_diffuse, dim=1)
        r_surface_multi_direct = torch.stack(r_surface_multi_direct, dim=1)
        r_surface_multi_diffuse = torch.stack(r_surface_multi_diffuse, dim=1)
        a_layer_multi_direct = torch.stack(a_layer_multi_direct, dim=1)
        a_layer_multi_diffuse = torch.stack(a_layer_multi_diffuse, dim=1)

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

        # Assign all 3 fluxes above the layer
        flux_down_direct = [flux_direct]
        flux_down_diffuse = [flux_diffuse]
        flux_up_diffuse = [flux_direct * upward_reflection_toa]

        # Will assign flux absorbed at each layer
        flux_absorbed = []

        for i in range(t_direct.shape[1]):

            flux_absorbed.append(flux_direct * a_layer_multi_direct[:,i] + 
                        flux_diffuse * a_layer_multi_diffuse[:,i])

            # Will want this later when incorporate surface interactions
            #flux_absorbed_surface = flux_direct * a_surface_multi_direct + \
            #flux_diffuse * a_surface_multi_diffuse

            flux_down_direct.append(flux_direct * t_direct[:,i])
            flux_down_diffuse.append(flux_direct * t_multi_direct[:,i] + 
                                    flux_diffuse * (t_diffuse[:,i] + t_multi_diffuse[:,i]))
            flux_up_diffuse.append(flux_direct * 
                                         r_surface_multi_direct[:,i] 
                                         + flux_diffuse * 
                                         r_surface_multi_diffuse[:,i])
            
            flux_direct = flux_down_direct[-1]
            flux_diffuse = flux_down_diffuse[-1]
        
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
    """ Computes full radiative transfer (direct and diffuse radiation)
    for an atmospheric column """

    def __init__(self, n_channel, n_constituent, device):
        super(FullNet, self).__init__()
        self.device = device
        self.n_channel = n_channel

        # Learns scalar diffuse zenith angle approximation
        self.mu_diffuse_net = nn.Linear(1,1,bias=False,device=device)
        torch.nn.init.uniform_(self.mu_diffuse_net.weight, a=0.4, b=0.6)
        self.sigmoid = nn.Sigmoid()

        # Learns decompositon of TOA radiation into channels
        self.spectral_net = nn.Linear(1,n_channel,bias=False,device=device)
        torch.nn.init.uniform_(self.spectral_net.weight, a=0.4, b=0.6)
        self.softmax = nn.Softmax(dim=-1)

        # Learns optical depth
        self.extinction_net = LayerDistributed(Extinction(n_channel,device))

        # Learns scattering
        self.scattering_net = LayerDistributed(Scattering(n_channel,
                                                          n_constituent,
                                                          device))
        # Computes multireflection among all layers
        self.multireflection_net = MultiReflection()

        # Propagation of radiation from TOA to surface
        self.propagation_net = Propagation(n_channel)

    def forward(self, x):
        x_layers, x_surface = x

        (mu_direct, 
         temperature_pressure_log_pressure, 
         constituents) = (x_layers[:,:,0:1], 
                          x_layers[:,:,1:4], 
                          x_layers[:,:,4:12])

        # Learn diffuse zenith angle
        one = torch.ones((mu_direct.shape[0],1),dtype=torch.float32,
                         device=self.device)
        mu_diffuse = self.sigmoid(self.mu_diffuse_net(one))
        mu_diffuse = mu_diffuse.repeat([1,mu_direct.shape[1]])
        mu_diffuse = torch.unsqueeze(mu_diffuse,dim=2)

        # Learn optical depth of each layer
        tau = self.extinction_net((temperature_pressure_log_pressure, 
                                   constituents))

        # Learn full set of radiative coefficients
        # from scattering
        layers = self.scattering_net((tau, mu_direct, mu_diffuse, 
                                                constituents))

        # Interaction (multireflection) among all layers
        (multireflected_layers, 
         upward_reflection_toa) = self.multireflection_net([layers,
                                                          x_surface])

        # Learn decomposition of flux into spectral channels at TOA
        flux_direct = self.softmax(self.spectral_net(one))
        flux_diffuse = torch.zeros((mu_direct.shape[0], self.n_channel),
                                         dtype=torch.float32,
                                         device=self.device)
        input_flux = [flux_direct, flux_diffuse]

        # Propagate flux through layers downward along spectral channels
        flux = self.propagation_net(multireflected_layers, 
                                    upward_reflection_toa,
                                    input_flux)

        (flux_down_direct, flux_down_diffuse, flux_up_diffuse, 
         flux_absorbed) = flux
        
        flux_down = flux_down_direct + flux_down_diffuse
        flux_up = flux_up_diffuse
        return [flux_down_direct, flux_down, flux_up, flux_absorbed]

def loss_weighted(y, y_pred, weight_profile):
    error = torch.mean(torch.square(weight_profile * (y_pred - y)), 
                       dim=(0,1), keepdim=False)
    return error

def loss_heating_rate_direct(y, y_pred, toa, delta_pressure):
    absorbed_true = y[:,:-1] - y[:,1:]
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
    loss = loss_heating_rate_direct(flux_down_direct_true, flux_down_direct_pred, toa, delta_pressure)
    return loss

def loss_heating_rate_full(flux_absorbed_true, flux_absorbed_pred, toa, delta_pressure):
    heat_true = absorbed_flux_to_heating_rate(flux_absorbed_true, delta_pressure)
    heat_pred = absorbed_flux_to_heating_rate(flux_absorbed_pred, delta_pressure)
    error = torch.sqrt(torch.mean(torch.square(toa * (heat_true - heat_pred)),
                                  dim=(0,1),keepdim=False))
    return error
def loss_heating_rate_full_wrapper(data, y_pred, weight_profile):
    _, _, toa, delta_pressure, _, flux_absorbed_true = data
    _, _, _, flux_absorbed_pred = y_pred
    loss = loss_heating_rate_full(flux_absorbed_true, flux_absorbed_pred, toa, delta_pressure)
    return loss

def loss_ukkonen_direct(y, y_pred, toa, delta_pressure, weight_profile):
    # y(n_examples, n_layers+1)
    loss_flux = loss_weighted (y, y_pred, weight_profile)
    hr_loss = loss_heating_rate_direct(y, y_pred, toa, delta_pressure)
    alpha   = 1.0e-4
    return alpha * hr_loss + (1.0 - alpha) * loss_flux

def loss_ukkonen_direct_wrapper(data, y_pred, weight_profile):
    _, y, x_toa, x_delta_pressure = data
    loss = loss_ukkonen_direct(y,y_pred,x_toa,x_delta_pressure,weight_profile)
    return loss

def loss_flux_direct_wrapper(data, y_pred, weight_profile):
    _, y, _, _ = data
    loss = loss_weighted(y,y_pred,weight_profile)
    return loss

def loss_flux_full_wrapper(data, y_pred, weight_profile):
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

def loss_henry_wrapper(data, y_pred, weight_profile):
    _, _, toa, delta_pressure, y_true, flux_absorbed_true = data
    loss = loss_henry(y_true, flux_absorbed_true, y_pred, toa, 
               delta_pressure, weight_profile)
    return loss


def train_loop(dataloader, model, optimizer, loss_function, weight_profile):
    """ Generic training loop """
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
    """ Generic testing / evaluation loop """
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

    x_layers, y, x_toa, x_delta_pressure = load_data_direct_pytorch(filename_training, n_channel)

    weight_profile = 1.0 / torch.mean(torch.from_numpy(y).float().to(device), dim=0, keepdim=True)

    tensorize = convert_to_tensor(device)
    train_dataset = torch.utils.data.TensorDataset(tensorize(x_layers), 
                                                   tensorize(y),
                                                   tensorize(x_toa),
                                                   tensorize(x_delta_pressure))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

    x_layers, y, x_toa, x_delta_pressure = load_data_direct_pytorch(filename_validation, n_channel)
    validation_dataset = torch.utils.data.TensorDataset(tensorize(x_layers), 
                                                   tensorize(y),
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
        train_loop(train_dataloader, model, optimizer, loss_ukkonen_direct_wrapper,weight_profile)
        loss = test_loop(validation_dataloader, model, loss_functions, loss_names, weight_profile)
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

    (x_layers, x_surface, x_toa, x_delta_pressure, 
     y_flux, y_flux_absorbed) = load_data_full_pytorch_2(filename_training, 
                                                         n_channel)

    weight_profile = get_weight_profile(y_flux, device)

    tensorize = convert_to_tensor(device)
    train_dataset = torch.utils.data.TensorDataset(tensorize(x_layers), 
                                                   tensorize(x_surface), 
                                                   tensorize(x_toa), 
                                                   tensorize(x_delta_pressure), 
                                                   tensorize(y_flux), 
                                                   tensorize(y_flux_absorbed))


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, 
                                                   shuffle=True)

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
                                                        shuffle=True)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    loss_functions = (loss_henry_wrapper, loss_heating_rate_full_wrapper,
                      loss_heating_rate_direct_full_wrapper, loss_flux_full_wrapper)
    loss_names = ("Loss", "Heating Rate Loss", "Direct Heating Rate Loss", "Flux Loss")

    if True:
        t = 0
    else:   
        t = 100
        checkpoint = torch.load(filename_full_model + str(t))
        print(f"Loaded Model: epoch = {t}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #epoch = checkpoint['epoch']
    while t < epochs:
        t += 1
        print(f"Epoch {t}\n-------------------------------")
        #with profiler.profile(with_stack=True, profile_memory=True) as prof:
        start.record()
        train_loop(train_dataloader, model, optimizer, loss_henry_wrapper, weight_profile)

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