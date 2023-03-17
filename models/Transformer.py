
import sys
from turtle import forward
sys.path.append("..") 
from models.modelbase import BaseModel
import torch.nn as nn
import torch
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import copy
import math
from torch.autograd import Variable

class Transformer(BaseModel):
    def __init__(self, n_dim, input_size, seq_len, N, heads, dropout, num_classes, train_dataset = None, test_dataset = None):
        super().__init__(train_dataset, test_dataset, test_dataset)

        self.embedding = nn.Linear(n_dim, input_size)
        self.encoder = Encoder(input_size, seq_len, N, heads, dropout)
        self.out = nn.Linear(input_size, num_classes) 
    
    def score(self, x, returnWeights=False):
        x = self.embedding(x)

        if(returnWeights):
            e_outputs, weights = self.encoder(x,returnWeights=returnWeights)
        else:
            e_outputs = self.encoder(x)
        
        e_outputs = torch.max(e_outputs.transpose(1, 2), dim = -1).values
        output = self.out(e_outputs)

        if returnWeights:
            return output, weights
        else:
            return output

    def forward(self, x):
        o = self.score(x)
        return F.log_softmax(o, dim=1)
    
    @torch.no_grad()
    def predict(self, x):
        self.eval()
        output = self.forward(x)
        return torch.argmax(output, dim = 1)

    def loss(self, y, y_hat, reduction='mean'):
        return F.nll_loss(y, y_hat, reduction=reduction)


def get_clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, input_size, seq_len, N, heads, dropout):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(input_size, seq_len, dropout = dropout)
        self.layers = get_clone(Encoderlayer(input_size, heads, dropout), N)
        self.norm = Norm(input_size)
    
    def forward(self, x, returnWeights = False):
        x = self.pe(x)
        for i in range(self.N):
            if i == 0 and returnWeights:
                x, weights = self.layers[i](x, returnWeights = returnWeights)
            else:
                x = self.layers[i](x)
        
        if (returnWeights):
            return self.norm(x), weights
        
        else:
            return self.norm(x)

    
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 100, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.shape[1]

        pe = Variable(self.pe[:, :seq_len], requires_grad = False)
        
        if x.is_cuda:
            pe.cuda()
        x = x + pe

        return self.dropout(x)

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.ones(self.size))

        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim = -1, keepdim = True)) \
            /(x.std(dim = -1, keepdim = True) + self.eps) + self.bias
        return norm 


def attention(q, k, v, d_k, mask=None, dropout=None,returnWeights=False):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)

    if(returnWeights):
        return output,scores

    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model//heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask = None, returnWeights = False):
        bs = q.shape[0]

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.k_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        if(returnWeights):
            scores,weights = attention(q, k, v, self.d_k, mask, self.dropout,returnWeights=returnWeights)
        else:
            scores = attention(q, k, v, self.d_k, mask, self.dropout)

        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)

        if(returnWeights):
            return output,weights
        
        else:
            return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=400, dropout = 0.1):
        super().__init__() 
    
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Encoderlayer(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()

        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout = dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, returnWeights=False):
        x2 = self.norm_1(x)
        if(returnWeights):
            attenOutput,attenWeights= self.attn(x2,x2,x2,returnWeights=returnWeights)
        else:
            attenOutput= self.attn(x2,x2,x2)

        x = x + self.dropout_1(attenOutput)
        x2 = self.norm_2(x)

        x = x + self.dropout_2(self.ff(x2))

        if(returnWeights):
            return x,attenWeights
        else:
            return x
