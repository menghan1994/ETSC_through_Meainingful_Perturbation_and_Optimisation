import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F


class BaseModel(pl.LightningModule):
    def __init__(self, train_dataset = None, test_dataset = None, val_dataset = None):
        super().__init__()
        self.lr = 0.001
        self.train_dataset = train_dataset
        self.val_dataset = test_dataset
        self.test_dataset = test_dataset
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, shuffle=True, num_workers = 4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32, shuffle=False, num_workers = 4)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32, shuffle=False, num_workers = 4)

    def training_step(self, batch, batch_nb, normalise = False):
        x = batch[0]['values'].to(torch.float)
        label = batch[1]

        output = self.forward(x)
        loss = self.loss(output, label)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_nb):
        x = batch[0]['values'].to(torch.float)
        label = batch[1]
        
        output = self.forward(x)
        val_loss = self.loss(output, label)
        
        pred_y = torch.argmax(output, dim=1)
        acc = torch.mean((pred_y == label).to(torch.float))

        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        self.log('acc', acc, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_nb):
        x = batch[0]['values'].to(torch.float)
        label = batch[1]
        
        output = self.forward(x)
        val_loss = self.loss(output, label)
        
        pred_y = torch.argmax(output, dim=1)
        acc = torch.mean((pred_y == label).to(torch.float))

        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        self.log('acc', acc, on_epoch=True, prog_bar=True)
        return val_loss

    @torch.no_grad()
    def prob(self, inputs):
        return torch.exp(self.forward(inputs))

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        output = self.forward(x)
        return torch.argmax(output, dim = 1)

    def loss(self, y, y_hat, reduction='mean'):
        return F.nll_loss(y, y_hat, reduction=reduction)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.8, patience=50, min_lr=1e-8)
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


    