

import torch.nn as nn
import torch 
from torch.autograd import Variable
import torch.nn.functional as F

from datasets.mask import MaskGenerator


class TemporalDecay(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TemporalDecay, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self,  x):
        return torch.exp(-F.relu(self.linear(x)))


class GRUIEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUIEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear_mu = nn.Linear(hidden_dim+input_dim, hidden_dim)
        self.linear_r = nn.Linear(hidden_dim+input_dim, hidden_dim)
        self.linear_h_hat = nn.Linear(hidden_dim+input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.timedecay = TemporalDecay(input_dim, hidden_dim)
    
    def forward(self, x, delta):
        
        h_t = Variable(torch.zeros((x.shape[0], self.hidden_dim))).to(torch.float).to(x.device)

        for t in range(x.shape[-2]):
            x_t = x[:, t, :]
            delta_t = delta[:, t, :]
            beta_t = self.timedecay(delta_t)
            h_t = beta_t * h_t

            temp_x = torch.cat((h_t, x_t), dim = 1)
            r_t = torch.sigmoid(self.dropout(self.linear_r(temp_x)))
            mu_t = torch.sigmoid(self.dropout(self.linear_mu(temp_x)))

            temp_r_for_h_hat= r_t * h_t
            temp_input_for_h_hat = torch.cat((temp_r_for_h_hat, x_t), dim=1)
            h_t_hat = torch.tanh(self.dropout(self.linear_h_hat(temp_input_for_h_hat)))
            
            h_t = (1 - mu_t) * h_t + mu_t * h_t_hat
        return h_t

class GRUIDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUIDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear_mu = nn.Linear(hidden_dim+input_dim, hidden_dim)
        self.linear_r = nn.Linear(hidden_dim+input_dim, hidden_dim)
        self.linear_h_hat = nn.Linear(hidden_dim+input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.timedecay = TemporalDecay(input_dim, hidden_dim)
        self.reg_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, delta):
        h_t = Variable(torch.zeros((x.shape[0], self.hidden_dim))).to(torch.float).to(x.device)

        complete_ts = []
        x_t = x
        for t in range(delta.shape[-2]):
            delta_t = delta[:, t, :]
            beta_t = self.timedecay(delta_t)
            h_t = beta_t * h_t

            temp_x = torch.cat((h_t, x_t), dim = 1)
            r_t = torch.sigmoid(self.dropout(self.linear_r(temp_x)))
            mu_t = torch.sigmoid(self.dropout(self.linear_mu(temp_x)))

            temp_r_for_h_hat= r_t * h_t
            temp_input_for_h_hat = torch.cat((temp_r_for_h_hat, x_t), dim=1)
            h_t_hat = torch.tanh(self.linear_h_hat(temp_input_for_h_hat))
            
            h_t = (1 - mu_t) * h_t + mu_t * h_t_hat
            x_t = torch.tanh(self.reg_out(h_t))
            complete_ts.append(x_t.unsqueeze(1))
        return torch.cat(complete_ts, dim = 1)

class E2EGanImputationGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(E2EGanImputationGenerator, self).__init__()
        self.encoder = GRUIEncoder(input_dim, hidden_dim)
        self.z = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_input = nn.Linear(hidden_dim, input_dim)
        self.decoder = GRUIDecoder(input_dim, hidden_dim)

    def forward(self, x, mask, delta):
        noise = torch.randn_like(x) * 0.01
        noise = noise.to(x.device)
        
        x = x * mask + noise

        z = self.z(self.encoder(x, delta))
        decoder_input = self.decoder_input(z)
        complete_ts = self.decoder(decoder_input, delta)
        return complete_ts

class E2EGanImputationDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(E2EGanImputationDiscriminator, self).__init__()
        self.encoder = GRUIEncoder(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask, delta):
        return self.output(torch.relu(self.encoder(x, delta)))
    
class E2EGanImputation(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(E2EGanImputation, self).__init__()
        self.generator = E2EGanImputationGenerator(input_dim, hidden_dim)
        self.discriminator = E2EGanImputationDiscriminator(input_dim, hidden_dim)
    

    def imputation(self, x, mask, delta):
        complete_ts = self.generator(x, mask, delta)
        x_complete = complete_ts * (1 - mask) + mask * x
        return x_complete

    

if __name__ == '__main__':
    maskgenerator = MaskGenerator((144, 9))
    encoder = E2EGanImputation(9, 32)

    x = torch.rand(10, 144, 9)
    deltas = []
    for _ in range(x.shape[0]):
        mask, delta = maskgenerator.random_mask(True)
        deltas.append(torch.from_numpy(delta).unsqueeze(0))
    delta = torch.cat(deltas, dim = 0)
    delta = delta.to(torch.float)
    x_o = x * mask
    x_o = x_o.to(torch.float)
    encoder(x_o, None, delta)












