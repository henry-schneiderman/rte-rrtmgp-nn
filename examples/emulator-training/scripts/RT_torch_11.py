import numpy as np
import torch
from torch import nn
import torch.autograd.profiler as profiler

from RT_data_hws import load_data_direct_pytorch, absorbed_flux_to_heating_rate

eps_1 = 0.0000001

class MLP(nn.Module):
    def __init__(self, n_hidden, n_input, n_output, device, lower=-0.1, upper=0.1):
        super(MLP, self).__init__()
        self.n_hidden = n_hidden
        self.n_outputs = n_output
        n_last = n_input
        self.hidden = nn.ModuleList()

        for n in n_hidden:
            mod = nn.Linear(n_last, n, bias=True,device=device)
            torch.nn.init.uniform_(mod.weight, a=lower, b=upper)
            torch.nn.init.uniform_(mod.bias, a=-0.1, b=0.1)
            self.hidden.append(mod)
            n_last = n
        self.activation = nn.ReLU()
        self.output = nn.Linear(n_last, n_output, bias=True, device=device)
        torch.nn.init.uniform_(self.output.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.output.bias, a=-0.1, b=0.1)

    def forward(self, x):
        for hidden in self.hidden:
            x = hidden(x)
            x = self.activation(x)
        return self.output(x)
    
class OpticalDepth(nn.Module):

    def __init__(self, n_channel, device):
        super(OpticalDepth, self).__init__()
        self.n_channel = n_channel
        self.net_lw  = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_iw  = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_h2o = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_o3  = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_co2 = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_u   = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_n2o = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_ch4 = nn.Linear(1,self.n_channel,bias=False,device=device)

        lower = 0.2
        upper = 1.8
        torch.nn.init.uniform_(self.net_lw.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_iw.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_h2o.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_o3.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_co2.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_u.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_n2o.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_ch4.weight, a=lower, b=upper)

        self.net_ke_h2o = MLP(n_hidden=(6,4,4),n_input=3,n_output=1,device=device)
        self.net_ke_o3  = MLP(n_hidden=(6,4,4),n_input=3,n_output=1,device=device)
        self.net_ke_co2 = MLP(n_hidden=(6,4,4),n_input=3,n_output=1,device=device)
        self.net_ke_u   = MLP(n_hidden=(6,4,4),n_input=3,n_output=1,device=device)
        self.net_ke_n2o = MLP(n_hidden=(6,4,4),n_input=3,n_output=1,device=device)
        self.net_ke_ch4 = MLP(n_hidden=(6,4,4),n_input=3,n_output=1,device=device)

        self.activation = nn.ReLU()

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
        
        self.activation_1 = nn.ReLU()
        self.activation_2 = nn.Sigmoid()

    def forward(self, x):
        t_p, c = x

        a = self.activation_1
        b = self.activation_2
        tau_lw  = a(self.net_lw (c[:,0:1]))
        tau_iw  = a(self.net_iw (c[:,1:2]))
        tau_h2o = a(self.net_h2o(c[:,2:3])) * self.filter_h2o * b(self.net_ke_h2o(t_p))
        tau_o3  = a(self.net_o3 (c[:,3:4])) * self.filter_o3  * b(self.net_ke_o3 (t_p))
        tau_co2 = a(self.net_co2(c[:,4:5])) * self.filter_co2 * b(self.net_ke_co2(t_p))
        tau_u   = a(self.net_u  (c[:,5:6])) * self.filter_u   * b(self.net_ke_u  (t_p))
        tau_n2o = a(self.net_n2o(c[:,6:7])) * self.filter_n2o * b(self.net_ke_n2o(t_p))
        tau_ch4 = a(self.net_ch4(c[:,7:8])) * self.filter_ch4 * b(self.net_ke_ch4(t_p))

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
    Input and output elements are: (samples, timesteps, input/output size)
    """
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

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
    """ Only Comutes Direct Transmission Coefficient """
    def __init__(self):
        super(LayerPropertiesDirect, self).__init__()

    def forward(self, x):
        mu, tau = x
        
        tau_total = torch.sum(tau,dim=2,keepdim=False)
        t_direct = torch.exp(-tau_total / (mu + eps_1))
        return t_direct

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
        self.optical_depth_net = TimeDistributed(OpticalDepth(n_channel,device))
        self.layer_properties_net = TimeDistributed(LayerPropertiesDirect())
        self.downward_propagate = DownwardPropagationDirect(n_channel)

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

def loss_ukkonen_direct(y, y_pred, toa, delta_pressure, weight_profile):
    # y(n_examples, n_layers+1)
    loss_flux = loss_weighted (y, y_pred, weight_profile)
    hr_loss = loss_heating_rate_direct(y, y_pred, toa, delta_pressure)
    alpha   = 1.0e-4
    return alpha * hr_loss + (1.0 - alpha) * loss_flux

def train_loop(dataloader, model, optimizer, weight_profile):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    loss_string = "Training Loss: "
    for batch, (X, y, toa, delta_pressure) in enumerate(dataloader):
        # Compute prediction and loss
        y_pred = model(X)
        loss = loss_ukkonen_direct(y, y_pred, toa, delta_pressure, weight_profile)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 20 == 0:
            loss = loss.item()
            #f"loss: {loss:>7f}")
            loss_string += f" {loss:>7f}"

    print (loss_string)


def test_loop(dataloader, model, weight_profile):
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
        for X, y, toa, delta_pressure in dataloader:
            y_pred = model(X)
            loss += loss_ukkonen_direct(y, y_pred, toa, delta_pressure, weight_profile).item()
            loss_heating_rate += loss_heating_rate_direct(y, y_pred, toa, delta_pressure).item()
            loss_flux += loss_weighted(y, y_pred, toa).item()

    loss /= num_batches
    loss_heating_rate /= num_batches
    loss_flux /= num_batches

    print(f"Test Error: \n Loss: {loss:>8f}\n Heating Rate Loss: {loss_heating_rate:>8f}")
    print(f" Flux Loss: {loss_flux:>8f}\n")
      
if __name__ == "__main__":

    print("Pytorch version:", torch.__version__)
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    datadir     = "/home/hws/tmp/"
    filename_training = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.2.nc"
    filename_validation = datadir + "/RADSCHEME_data_g224_CAMS_2014.2.nc"
    filename_testing = datadir + "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"

    batch_size = 2048
    n_channel = 30
    model = DirectDownwardNet(n_channel,device)
    model = model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters())

    epochs = 4000

    X_train, y_train, toa_train, delta_pressure_train = load_data_direct_pytorch(filename_training, n_channel)
    weight_profile = 1.0 / torch.mean(torch.from_numpy(y_train).float().to(device), dim=0, keepdim=True)
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float().to(device), 
                                                   torch.from_numpy(y_train).float().to(device),
                                                   torch.from_numpy(toa_train).float().to(device),
                                                   torch.from_numpy(delta_pressure_train).float().to(device))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

    X_valid, y_valid, toa_valid, delta_pressure_valid = load_data_direct_pytorch(filename_validation, n_channel)
    validation_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_valid).float().to(device), 
                                                        torch.from_numpy(y_valid).float().to(device),
                                                        torch.from_numpy(toa_valid).float().to(device),
                                                        torch.from_numpy(delta_pressure_valid).float().to(device))

    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size, shuffle=True)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        #with profiler.profile(with_stack=True, profile_memory=True) as prof:
        train_loop(train_dataloader, model, optimizer, weight_profile)
        test_loop(validation_dataloader, model, weight_profile)
    print("Done!")
    #print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
