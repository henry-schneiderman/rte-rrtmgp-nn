import torch
from torch import nn


class BottomUp(nn.Module):


    def _compute_surface_reflection(self,r,t):
        """
        In bottom up order compute cumulative surface reflection
        and denominator factor
        """

        rs = []
        last_rs = r[:,-1,:]  # reflectance of original surface is unchanged
                                # n-th entry
        rs.append(last_rs)
        d = []

        #for l in range(r.shape[1]-1, -1, -1):
        for l in reversed(torch.arange(start=0, end=r.shape[1])):
            dd = 1 - last_rs * r[:,l,:]
            d.append(dd)
            tmp = r[:,l,:] + last_rs * t[:,l,:] * t[:,l,:]
            last_rs = tmp / dd
            rs.append(last_rs)

        rs = torch.stack(rs,dim=1)  
        d = torch.stack(d,dim=1)  

        rs = torch.flip(rs, dims=(1,))  # n+1 values: 0 .. n
        d = torch.flip(d, dims=(1,)) # n values: 0 .. n-1

        return rs, d

    def _compute_upward_flux (self, s_multi_up, s_multi_down_up, d, t):
        """ s: 1..n """
        flux = torch.zeros()
        result = []

        for l in reversed(torch.arange(s_multi_up.shape[1])):
            flux += s_multi_up[:,l,:] + s_multi_down_up[:,l,:]
            result.append(flux)
            flux = flux * t[:,l,:] / d[:,l,:]  # could be t_multi

        result = torch.stack(result, dim=1)
        result = torch.flip(result, dims=(1,))

        return result
    
    def _compute_downward_flux (self, s_multi_down, s_multi_up_down, t_multi):
        """ s: 1..n """
        flux = torch.zeros()
        result = []

        for l in torch.arange(s_multi_down.shape[1]):
            flux += s_multi_down[:,l,:] + s_multi_up_down[:,l,:]
            result.append(flux)
            flux = flux * t_multi[:,l,:]  # could be t_multi

        result = torch.stack(result, dim=1)

        return result

    def _adding_doubling (self, a, r, t, s):
        """
        Inputs:
            Dimensions[examples, layers, channels]
            n + 1 layers including surfaces: 0 .. n
        """
        rs, d = self._compute_surface_reflection(r,t)
        # n layers (does not include above top layer (layer=0))
        tmp = rs[:,1,:]*r[:,:-1,:]
        s_multi_up = s[:,1:,:] * (1.0 + tmp / (1 + tmp)) # n-1 values: 1..n
        s_multi_down_up = s[:,:-1,:] * rs[:,1,:] / (1 + tmp) # n-1 values: 1..n

        # used in downward propagation
        a_multi = a[:,:-1,:] * (1.0 + t[:,:-1,:] * rs[:,1:,:] / (1 + tmp)) # n-1 values: 0..n-1
        t_multi = t[:,:-1,:] / (1.0 + tmp) # n-1 values: 0..n-1

        flux_up = self._compute_upward_flux (s_multi_up, s_multi_down_up, d, t)

        absorbed_flux_up = a[:,:-1,:] * flux_up # n values: 0..n-1

        s_multi_down = s[:,:-1,:] * (1.0 + tmp / (1 - tmp))
        s_multi_up_down = s[:,1:,:] * r[:,:-1,:] / (1 - tmp)

        flux_down = self._compute_downward_flux (s_multi_down, s_multi_up_down, t_multi)
        absorbed_flux_down = a_multi * flux_down

        absorbed_flux = absorbed_flux_down + absorbed_flux_up