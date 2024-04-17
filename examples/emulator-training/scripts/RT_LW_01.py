import torch
from torch import nn


class BottomUp(nn.Module):


    def _compute_surface_reflection(self,r,t):
        """

        Bottom up computation of cumulative surface reflection
        and denominator factor

        d index corresponds to virtual surface (n elements)
        rs index corresponds to virtual surface (n+1 elements)
        """

        rs = []
        last_rs = r[:,-1,:]  # reflectance of original surface is unchanged

        rs.append(last_rs)
        d = []

        # Start at bottom of the atmosphere (layer = n-1)
        for l in reversed(torch.arange(start=0, end=r.shape[1]-1)):
            dd = 1.0 / (1.0 - last_rs * r[:,l,:])
            d.append(dd)
            last_rs = r[:,l,:] + last_rs * t[:,l,:] * t[:,l,:] * dd
            rs.append(last_rs)

        rs = torch.stack(rs,dim=1)  
        d = torch.stack(d,dim=1)  

        rs = torch.flip(rs, dims=(1,))  # n+1 values: 0 .. n (includes surface)
        d = torch.flip(d, dims=(1,)) # n values: 0 .. n-1 (no surface)

        return rs, d
    
    def _compute_top_reflection(self,r,t):
        """
        In top down order compute cumulative upper layer reflection
        and denominator factor
        """

        # Start at top of the atmosphere 
        rt = []
        last_rt = r[:,0,:]  # reflectance of top surface is unchanged
        rt.append(last_rt)
        d = []
        d.append(torch.ones())

        for l in torch.arange(start=1, end=r.shape[1]-1):
            dd = 1.0 / (1.0 - last_rt * r[:,l,:])
            d.append(dd)
            last_rt = r[:,l,:] + last_rt * t[:,l,:] * t[:,l,:] * dd
            rt.append(last_rt)

        dd = 1.0 / (1.0 - last_rt * r[:,-1,:])

        rt = torch.stack(rt,dim=1)  # n values: (excludes surface)
        d = torch.stack(d,dim=1)  # n + 1 values

        return rt, d


    
    def _compute_upward_flux (self, s_up, rt, dt, t, a):
        """ 
        Input:
            s_up: n+1 elements, from layer+surface
            
        Output:
            flux_up: n+1 elements from layer
            absorbed_flux: n elements: into surface + layers (surface is zero since going up)
        """
        flux = torch.zeros()
        flux_up = []
        absorbed_flux = []
        absorbed_flux.append(torch.zeros())

        # from n to 1
        for l in reversed(torch.arange(2,s_up.shape[1])):
            flux += s_up[:,l,:] 
            flux_up.append(flux)
            a_multi = a[:,l-1,:] * (1.0 + t[:,l-1,:] * rt[:,l-2,:] * dt[:,l-1,:])
            absorbed_flux.append(flux * a_multi) # absorbed at l-1
            # propagate into next layer
            flux = flux * t[:,l-1,:] * dt[:,l-1,:]
        flux += s_up[:,1,:]
        flux_up.append(flux)
        absorbed_flux.append(flux * a[:,0,:])
        flux = flux * t[:,0,:]

        flux += s_up[:,0,:] # from layer zero toward upper atmosphere
        flux_up.append(flux)

        flux_up = torch.stack(flux_up, dim=1) # from layer, n+1 values
        flux_up = torch.flip(flux_up, dims=(1,))

        absorbed_flux = torch.stack(absorbed_flux, dim=1) # n+1 values for n layers
        absorbed_flux = torch.flip(absorbed_flux, dims=(1,)) # since includes surface

        return absorbed_flux, flux_up
    
    def _compute_downward_flux (self, s_down, rs, ds, t, a):
        """ 
        Input as flux from layer; n values from n layers (no surface)
        Output is flux, absorbed_flux into layer; n+1 values = n layers + surface
        """

        flux = torch.zeros()
        flux_down = []
        flux_down.append(flux)
        absorbed_flux = []
        absorbed_flux.append(torch.zeros())

        for l in torch.arange(0, s_down.shape[1]-1):
            flux += s_down[:,l,:]
            flux_down.append(flux)
            a_multi = a[:,l+1,:] * (1 + rs[:,l+2,:] * t[:,l+1,:] * ds[:,l+1,:])
            absorbed_flux.append(a_multi * flux)
            flux = flux * t[:,l+1,:] * ds[:,l+1,:]

        flux += s_down[:,-1,:]
        flux_down.append(flux)
        absorbed_flux.append(a[:,-1,:] * flux)

        absorbed_flux = torch.stack(absorbed_flux, dim=1)
        flux_down = torch.stack(flux_down, dim=1)

        # n+1 output values
        return absorbed_flux, flux_down

    def _adding_doubling (self, a, r, t, s):
        """
        All inputs:
            Dimensions[examples, layers, channels]
            n + 1 layers including surface: 0 .. n
        """

        # Bottom up cumulative surface reflection
        rs, ds = self._compute_surface_reflection(r,t)

        ### Downward sources
        s_multi_down = s[:,:-1,:] * ds  # n values, flow from layer
        s_multi_up_down = s[:,1:,:] * r[:,:-1,:] * ds

        s_down = s_multi_down + s_multi_up_down # index corresponds to from layer

        absorbed_flux_down, flux_down = self._compute_downward_flux (s_down, rs, ds, t, a)

        # Top down cumulative top layer reflection
        rt, dt = self._compute_top_reflection(r,t)

        ### Upward sources
        s_multi_up = s * dt    # from origin surface going up
        s_multi_down_up = s[:,:-1,:] * r[:,1:,:] * dt[:,1:,:] # from origin surface going down then up
        s_up = s_multi_up + torch.cat([torch.zeros(), s_multi_down_up], axis=1) 

        absorbed_flux_up, flux_up = self._compute_upward_flux (s_up, rt, dt, t, a)

        absorbed_flux = absorbed_flux_down + absorbed_flux_up

        return flux_down, flux_up, absorbed_flux