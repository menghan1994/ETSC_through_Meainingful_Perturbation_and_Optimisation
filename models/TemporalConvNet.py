
import sys
from turtle import forward
sys.path.append("..") 
from models.modelbase import BaseModel
import torch.nn as nn
import torch
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import pytorch_lightning as pl

class TCN(BaseModel):

    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, train_dataset = None, test_dataset = None):
        super().__init__(train_dataset, test_dataset, test_dataset)
        self.tcn = TemporalCovNet(input_size, num_channels, kernel_size = kernel_size, dropout = dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        inputs = torch.transpose(inputs, 1, 2)
        y1 = self.tcn(inputs)
        out = self.linear(y1[:, :, -1])
        return F.log_softmax(out, dim=1)

class TemporalCovNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size = 2, dropout = 0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        x = x.to(torch.float)
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)