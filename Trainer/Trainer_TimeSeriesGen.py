from Trainer.Trainer import Trainer
import torch 
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

class TimeSeriesGEN_Trainer(Trainer):
    def __init__(self, optim_params, logs_params, modelsave_params):
        super().__init__(optim_params, logs_params, modelsave_params)

   
    def train_epoch(self, model):
        G_losses = []
        D_losses = []
        reconstrucion_losses = []
        for batch, _ in model.train_dataloader():
            x, mask = batch['values'], batch['masks']

            x_o = x * mask
            x = x.to(torch.float).to(self.device)
            x_o = x_o.to(torch.float).to(self.device)
            mask = mask.to(torch.float).to(self.device)

            
            for _ in range(1):
                self.optim_D.zero_grad()
                D_real = model.descriminator(x)
                D_r_loss = -torch.mean(D_real)

                out = model.generator(x_o)
                fake_img = x * mask + out * (1 - mask)
                D_fake = model.descriminator(fake_img)
                D_f_loss = torch.mean(D_fake)

                D_loss = D_r_loss + D_f_loss + 10 * model.gradient_pently(x, fake_img)

                D_loss.backward()

                self.optim_D.step()
                D_losses.append(D_loss.item())
            
            for _ in range(1):
                self.optim_G.zero_grad()

                out = model.generator(x_o)
                fake_img = x * mask + out * (1 - mask)

                D_output = model.descriminator(fake_img)
                G_loss = -torch.mean(D_output)

                rec_loss = model.generator._get_loss(x, out, 1-mask)
                G_loss = 0.1*G_loss + 0.9 * rec_loss

                G_loss.backward()
                self.optim_G.step()
                G_losses.append(G_loss.item())

                reconstrucion_losses.append(rec_loss.item())

        return np.mean(G_losses), np.mean(D_losses), np.mean(reconstrucion_losses)
    
    def optim_init(self, model):
        self.optim_G = torch.optim.Adam(model.generator.parameters(),
            lr=self.lr)
        self.optim_D = torch.optim.Adam(model.descriminator.parameters(), 
            lr = self.lr)
    
    def fit(self, model):
        model = model.to(self.device)
        self.optim_init(model)

        writer = SummaryWriter(self.logs_dir)

        for epoch in tqdm(range(self.max_epoches)):

            model.train()
            train_G_loss, train_D_loss, reconstrucion_loss  = self.train_epoch(model)

            if epoch % 1 == 0:
                print('Epoach {}, Training....., G_loss:{}'.format(epoch, train_G_loss))
                print('Epoach {}, Training....., D_loss:{}'.format(epoch, train_D_loss))
                print('Epoach {}, Training....., reconstrucion_loss:{}'.format(epoch, reconstrucion_loss))

            writer.add_scalar('loss/train_G_loss', train_G_loss, epoch)
            writer.add_scalar('loss/train_D_loss', train_D_loss, epoch)
            writer.add_scalar('loss/reconstrucion_loss', reconstrucion_loss, epoch)

            writer.add_scalars(
                'loss/train_loss', {'train_G_loss': train_G_loss, 'test_D_loss': train_D_loss, 'reconstrucion_loss': reconstrucion_loss}, epoch)

            if not os.path.exists(self.model_save_dir):
                os.makedirs(self.model_save_dir)
            self.save_model(model, label='best.pth')
