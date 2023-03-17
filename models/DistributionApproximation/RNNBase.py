
import torch.nn as nn

class TimeSeriesRNNDiscriminator(nn.Module):
    def __init__(self, latent_dim, input_dim = 28, sequence_length = 28):
        super(TimeSeriesRNNDiscriminator, self).__init__()
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