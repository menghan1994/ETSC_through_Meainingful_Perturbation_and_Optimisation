import torch.nn as nn
import torch
import sys
sys.path.append("../..") 
from models.modelbase import BaseModel
import torch.nn.functional as F

    
class TimeSeriesGen_GAN(BaseModel):
    def __init__(self, params, train_dataset = None, test_dataset = None, val_dataset = None):
        super().__init__(train_dataset, test_dataset, val_dataset)

        # self.generator = TransformerBasedGEN(params  = params)
        self.generator = BiRNNBasedGEN(params  = params)
        self.descriminator = TimeSeriesRNNDiscriminator(params  = params)

    # @torch.backends.cudnn.flags(enabled=False)
    def gradient_pently(self, real_x, fake_x):
        t = torch.rand(real_x.shape[0], 1, 1).to(self.device)
        mid = t * real_x + (1 - t) * fake_x
        mid.requires_grad_()
        pred = self.descriminator(mid)
        grads = torch.autograd.grad(outputs=pred, inputs=mid,
                                    grad_outputs=torch.ones_like(pred),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        gp = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()
        return gp

class TimeSeriesRNNDiscriminator(nn.Module):
    def __init__(self, params):
        super(TimeSeriesRNNDiscriminator, self).__init__()
        
        latent_dim = params['latent_dim']
        input_dim = params['input_dim']
        sequence_length = params['sequence_length']
        self.phi_x = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )

        self.bidirec_GRU = nn.GRU(
                                    input_size = latent_dim, 
                                    hidden_size = latent_dim, 
                                    num_layers = 2, 
                                    batch_first = True,
                                    dropout = 0.05, 
                                    bidirectional = True
                                )
        
        self.out1 =  nn.Linear(latent_dim*2, 1)
        self.out2 =  nn.Linear(sequence_length, 1)
    # forward method
    def forward(self, x):
        out = self.phi_x(x)
        out, _ = self.bidirec_GRU(out)
        out = F.relu(self.out1(out)).reshape(out.shape[0], -1)
        return self.out2(out)
    

class GENBase(nn.Module):
    def __init__(self):
        super(GENBase, self).__init__()
    

    @torch.no_grad()
    def sample(self, x_o, mask):
        output = self.forward(x_o)
        complete_ts = mask * x_o + (1 - mask) * output
        return complete_ts

    def _get_loss(self, out, x, mask):
        target_mask = 1 - mask
        total_loss = nn.MSELoss(reduction = 'sum')(out * target_mask, x* target_mask) / (target_mask.sum())
        return total_loss
    
class BiRNNBasedGEN(GENBase):
    def __init__(self, params):
        super(BiRNNBasedGEN, self).__init__()


        self.lr = params['lr']

        self.dropout = nn.Dropout(0.5)
        self.hidden_dim = params['latent_dim']
        self.input_dim = params['input_dim']

        self.phi_x = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU()
        )


        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8, batch_first = True, dropout=0.5)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.encoder = nn.GRU(      input_size = self.hidden_dim, 
                                    hidden_size = self.hidden_dim, 
                                    num_layers = 2, 
                                    batch_first = True,
                                    dropout = 0.5, 
                                    bidirectional = True
                                )

        self.outputlayer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
            nn.Tanh()
        )

        self.rec_loss = nn.MSELoss(reduction='none')

    def forward(self, x_o, x_u = None, mask = None):
        out = self.phi_x(x_o)
        out, _ = self.encoder(out)
        out = self.dropout(out)
        out = self.outputlayer(out)
        return out
    
class TransformerBasedGEN(GENBase):
    def __init__(self, params):
        super(TransformerBasedGEN, self).__init__()

        self.lr = params['lr']

        self.dropout = nn.Dropout(0.5)
        self.hidden_dim = params['latent_dim']
        self.input_dim = params['input_dim']

        self.phi_x = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8, batch_first = True, dropout=0.5)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.outputlayer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
            nn.Tanh()
        )

        self.rec_loss = nn.MSELoss(reduction='none')

    def forward(self, x_o, x_u = None, mask = None):
        out = self.phi_x(x_o)
        out = self.encoder(out)
        out = self.dropout(out)
        out = self.outputlayer(out)
        return out

