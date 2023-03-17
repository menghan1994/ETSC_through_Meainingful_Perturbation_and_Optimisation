import sys
from turtle import forward
sys.path.append("..") 
from models.modelbase import BaseModel
import torch.nn as nn
import torch.nn.functional as F
import torch

class TSPredModel(BaseModel):
    def __init__(self, input_dim = 28, hidden_size = 128, num_classes = 10, train_dataset = None, test_dataset = None, val_dataset = None):
        super().__init__(train_dataset, test_dataset, val_dataset)
        self.hidden_size = hidden_size
        self.classes = num_classes
        self.drop = nn.Dropout(0.05)
        self.rnn = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.drop(x)
        output, _ = self.rnn(x)
        output = self.drop(output)
        output=output[:,-1,:]
        out = self.fc(output)
        return F.log_softmax(out, dim=1)