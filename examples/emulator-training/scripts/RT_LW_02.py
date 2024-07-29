# Fixed problem with RT_LW_01 to use multi-reflection coeffs on sources

import torch
from torch import nn


class BottomUp(nn.Module):


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
            n + 1 layers including surface: 0 .. n
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