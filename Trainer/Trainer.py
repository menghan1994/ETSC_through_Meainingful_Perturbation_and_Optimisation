from importlib.machinery import OPTIMIZED_BYTECODE_SUFFIXES
from pickletools import optimize
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn
import os
import torch


class Trainer():
    def __init__(self, optim_params, logs_params, modelsave_params, device = 'cpu'):
        self.lr = optim_params['lr']
        self.max_epoches = optim_params['max_epoches']
        if 'early_stop' in optim_params.keys():
            self.early_stop = optim_params['early_stop']

        self.logs_dir = logs_params['log_dir']
        self.model_save_dir = modelsave_params['model_save_dir']
        self.device = device
    def train_epoch(self, model):

        losses = []
        for batch in model.train_dataloader():
            x, y = batch
            x = x.to(torch.float).to(self.device)
            y = y.to(torch.int64).to(self.device)

            output = model.forward(x)
            loss = model.loss(output, y)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            losses.append(loss.item())
        return np.mean(losses)

    @torch.no_grad()
    def test_epoch(self, model):
        losses = []
        accs = []
        
        for batch in model.val_dataloader():
            x, y = batch
            x = x.to(torch.float).to(self.device)
            y = y.to(torch.int64).to(self.device)
            
            output = model.forward(x)
            pred_y = torch.argmax(output, dim=1)
            acc = torch.sum(pred_y == y)
            accs.append(acc.item())

            loss = model.loss(output, y) * x.shape[0]
            losses.append(loss.item())

        return np.sum(losses)/len(model.val_dataset), np.sum(accs)/len(model.val_dataset)

    def fit(self, model):
        model = model.to(self.device)
        self.optim_init(model)

        writer = SummaryWriter(self.logs_dir)
        best_loss = float('inf')
        early_stop = 0
        for epoch in tqdm(range(self.max_epoches)):

            model.train()
            train_loss = self.train_epoch(model)
            model.eval()
            test_loss, test_acc = self.test_epoch(model)

            if epoch % 50 == 0:
                print('Epoach {}, Training....., loss:{}'.format(epoch, train_loss))
                print('Epoach {}, Test....., loss:{}'.format(epoch, test_loss))
                print('Epoach {}, TestAcc....., loss:{}'.format(epoch, test_acc))

            writer.add_scalar('loss/train_loss', train_loss, epoch)
            writer.add_scalar('loss/test_loss', test_loss, epoch)
            writer.add_scalar('acc/test_acc', test_acc, epoch)

            if test_loss < best_loss:
                best_loss = test_loss
                if not os.path.exists(self.model_save_dir):
                    os.makedirs(self.model_save_dir)
                self.save_model(model, label='best.pth')
                early_stop = 0
            else:
                early_stop += 1
            
            if early_stop > self.early_stop:
                break


    def save_model(self, model, label):
        model_save_name = os.path.join(self.model_save_dir, label)
        torch.save(model.state_dict(), model_save_name)

    def optim_init(self, model):
        self.optim = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=0.0001)

    @torch.no_grad()
    def eval(self, model):
        model_save_name = os.path.join(self.model_save_dir, 'best.pth')
        model.load_state_dict(torch.load(model_save_name))
        model = model.to(self.device)
        model.device = self.device

        accs = []
        for batch in model.val_dataloader():
            x, y = batch
            x = x.to(torch.float).to(self.device)
            y = y.to(torch.int64).to(self.device)
            pred_y = model.predict(x)
            acc = torch.sum(pred_y == y)
            accs.append(acc.item())
        return np.sum(accs)/len(model.val_dataset)