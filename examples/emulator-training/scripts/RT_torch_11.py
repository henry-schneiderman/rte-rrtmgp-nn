import numpy as np
import torch
from torch import nn

from RT_data_hws import load_data_direct_pytorch, absorbed_flux_to_heating_rate

eps_1 = 0.0000001

class MLP(nn.Module):
    def __init__(self, n_hidden, n_output, device):
        super(MLP, self).__init__()
        self.n_hidden = n_hidden
        self.n_outputs = n_output
        self.hidden = nn.ModuleList([nn.LazyLinear(n, bias=True,device=device) for n in n_hidden])
        self.activation = nn.ReLU()
        self.output = nn.LazyLinear(n_output, bias=True, device=device)

    def forward(self, x):
        for hidden in self.hidden:
            x = hidden(x)
            x = self.activation(x)
        return self.output(x)
    
class OpticalDepth(nn.Module):

    def __init__(self, n_channel, device):
        super(OpticalDepth, self).__init__()
        self.n_channel = n_channel
        self.net_lw  = nn.LazyLinear(self.n_channel,bias=False,device=device)
        self.net_iw  = nn.LazyLinear(self.n_channel,bias=False,device=device)
        self.net_h2o = nn.LazyLinear(self.n_channel,bias=False,device=device)
        self.net_o3  = nn.LazyLinear(self.n_channel,bias=False,device=device)
        self.net_co2 = nn.LazyLinear(self.n_channel,bias=False,device=device)
        self.net_u   = nn.LazyLinear(self.n_channel,bias=False,device=device)
        self.net_n2o = nn.LazyLinear(self.n_channel,bias=False,device=device)
        self.net_ch4 = nn.LazyLinear(self.n_channel,bias=False,device=device)

        self.net_ke_h2o = MLP(n_hidden=(6,4,4),n_output=1,device=device)
        self.net_ke_o3  = MLP(n_hidden=(6,4,4),n_output=1,device=device)
        self.net_ke_co2 = MLP(n_hidden=(6,4,4),n_output=1,device=device)
        self.net_ke_u   = MLP(n_hidden=(6,4,4),n_output=1,device=device)
        self.net_ke_n2o = MLP(n_hidden=(6,4,4),n_output=1,device=device)
        self.net_ke_ch4 = MLP(n_hidden=(6,4,4),n_output=1,device=device)

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
        
        self.activation = nn.ReLU()

    def forward(self, x):
        t_p, c = x

        a = self.activation
        tau_lw  = a(self.net_lw (c[:,0:1]))
        tau_iw  = a(self.net_iw (c[:,1:2]))
        tau_h2o = a(self.net_h2o(c[:,2:3])) * self.filter_h2o * a(self.net_ke_h2o(t_p))
        tau_o3  = a(self.net_o3 (c[:,3:4])) * self.filter_o3  * a(self.net_ke_o3 (t_p))
        tau_co2 = a(self.net_co2(c[:,4:5])) * self.filter_co2 * a(self.net_ke_co2(t_p))
        tau_u   = a(self.net_u  (c[:,6:7])) * self.filter_u   * a(self.net_ke_u  (t_p))
        tau_n2o = a(self.net_n2o(c[:,7:8])) * self.filter_n2o * a(self.net_ke_n2o(t_p))
        tau_ch4 = a(self.net_ch4(c[:,8:9])) * self.filter_ch4 * a(self.net_ke_ch4(t_p))

        return [tau_lw, tau_iw, tau_h2o, tau_o3, tau_co2, tau_u, tau_n2o, tau_ch4]
    
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
        tau_lw, tau_iw, tau_h2o, tau_o3, tau_co2, tau_u, tau_n2o, tau_ch4 = tau
        tau_total = tau_lw + tau_iw + tau_h2o + tau_o3 + tau_co2 + tau_u + tau_n2o + tau_ch4
        t_direct = torch.exp(-tau_total / (mu + eps_1))
        return t_direct

class DownwardPropagationDirect(nn.Module):
    def __init__(self,n_channel):
        super(DownwardPropagationDirect, self).__init__()
        self.n_channel = n_channel

    def forward(self, x):
        flux_down_above_direct, t_direct = x
        flux_down_above_direct_channels = flux_down_above_direct.repeat(1,n_channel)
        flux_down_direct_channels = []
        flux = flux_down_above_direct_channels
        for i in range(t_direct.size(1)):
            flux = flux * t_direct[:,i,:]
            flux_down_direct_channels.append(flux)

        output = torch.stack(flux_down_direct_channels,0)
        output = torch.transpose(output,0,1)
        return output


class DirectDownwardNet(nn.Module):

    def __init__(self, n_channel, device):
        super(DirectDownwardNet, self).__init__()
        self.device = device
        self.spectral_net = nn.Linear(1,n_channel,bias=False,device=device)
        self.softmax = nn.Softmax()
        self.optical_depth_net = TimeDistributed(OpticalDepth(n_channel,device))
        self.layer_properties_net = TimeDistributed(LayerPropertiesDirect())
        self.downward_propagate = DownwardPropagationDirect(n_channel)

    def forward(self, x):
        mu, temperature_pressure, constituents = x[:,:,0:1], x[:,:,1:4], x[:,:,4:]
        one = torch.unsqueeze(torch.ones((mu.shape[0]),dtype=torch.float32,device=self.device), 1)
        flux_down_above_direct = self.softmax(self.spectral_net(one))
        tau = self.optical_depth_net((temperature_pressure, constituents))
        t_direct = self.layer_properties_net((mu, tau))
        flux_down_direct_channels = self.downward_propagate(flux_down_above_direct,t_direct)
        flux_down_direct = torch.sum(flux_down_direct_channels,dim=2,keepdim=False)
        return flux_down_direct
    
def weighted_loss(y_true, y_pred, weight_profile):
    error = torch.mean(torch.square(weight_profile * (y_pred - y_true)), dim=(0,1), keepdim=False)
    return error

def heating_rate_loss(absorbed_true, absorbed_pred, toa, delta_pressure):
    heat_true = absorbed_flux_to_heating_rate(absorbed_true, delta_pressure)
    heat_pred = absorbed_flux_to_heating_rate(absorbed_pred, delta_pressure)
    error = torch.sqrt(torch.mean(torch.square(toa * (heat_true - heat_pred)),dim=(0,1),keepdim=False))
    return error

def heating_rate_loss_direct(y_true, y_pred, toa, delta_pressure):
    absorbed_true = y_true[:,:-1] - y_true[:,1:]
    absorbed_pred = y_pred[:,:-1] - y_pred[:,1:]
    error = heating_rate_loss(absorbed_true, absorbed_pred, toa, delta_pressure)
    return error

def ukkonen_loss_direct(y_pred, ground_truth):
    # y(n_examples, n_layers+1)
    y_true, toa, delta_pressure = ground_truth[:,:,0], ground_truth[:,:,1], ground_truth[:,:,2]
    weight_profile = 1.0 / torch.mean(y_true, dim=0, keepdim=True)
    flux_loss = weighted_loss(y_true, y_pred, weight_profile)
    hr_loss = heating_rate_loss_direct(y_true, y_pred, toa, delta_pressure)
    alpha   = 1.0e-4
    return alpha * hr_loss + (1.0 - alpha) * flux_loss

class AtmosphereData(torch.utils.data.Dataset):
    def __init__(self, file_name, n_channel):
        self.x, self.y = load_data_direct_pytorch(file_name, n_channel)
    def __len__(self):
        return self.x[0].shape[0]
    def __get_item__(self,idx):
        x = []
        for xx in self.x:
            x.append(xx[idx])
        y = []
        for yy in self.y:
            y.append(yy[idx])
        return x, y


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0.0
    
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches

    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

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

    loss_fn = ukkonen_loss_direct

    epochs = 10

    x_train, y_train = load_data_direct_pytorch(filename_training, n_channel)
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float().to(device), torch.from_numpy(y_train).float().to(device))
    #training_data = AtmosphereData(filename_training, n_channel)
    #train_dataset = torch.utils.data.TensorDataset(training_data)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

    x_valid, y_valid = load_data_direct_pytorch(filename_validation, n_channel)
    validation_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_valid).float().to(device), torch.from_numpy(y_valid).float().to(device))
    #validation_data = AtmosphereData(filename_validation, n_channel)
    #validation_dataset = torch.utils.data.TensorDataset(validation_data)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size, shuffle=True)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(validation_dataloader, model, loss_fn)
    print("Done!")
