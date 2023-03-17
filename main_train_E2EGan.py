from datasets.MultivariateUEA import MultivariateUEA_train, MultivariateUEA_test
from models.E2EGan.E2EGanImputation import E2EGanImputation
import pandas as pd
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

device = 'cuda:1'


def pretrain(model, dataloader, optimizer, epoches = 10):
    
    for epoch in range(epoches):
        losses= 0.0
        for bs, _ in dataloader:
            x = bs['values'].to(torch.float).to(device)
            masks = bs['masks'].to(torch.float).to(device)
            deltas = bs['deltas'].to(torch.float).to(device)
            generated_ts = model.generator(x, masks, deltas)
            loss = torch.sum((x - generated_ts)**2 * masks/torch.sum(masks))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
    return model

def optim_generator(model, dataloader, optimizer, epoches = 1):
    
    mse_lossse = []
    D_f_losses = []
    for epoch in range(epoches):
        for bs, _ in dataloader:
            x = bs['values'].to(torch.float).to(device)
            masks = bs['masks'].to(torch.float).to(device)
            deltas = bs['deltas'].to(torch.float).to(device)
            generated_ts = model.generator(x, masks, deltas)
            D_fake = model.discriminator(generated_ts, masks, deltas)
            D_f_loss = torch.mean(D_fake)
            mse_loss = torch.sum((x - generated_ts)**2 * masks/torch.sum(masks))
            loss =  mse_loss - 0.01 * D_f_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mse_lossse.append(mse_loss.item())
            D_f_losses.append(D_f_loss.item())
    return np.mean(mse_lossse), np.mean(D_f_losses)

def optim_discriminator(model, dataloader, optimizer, epoches = 1):

    D_f_losses = []
    D_r_losses = []
    for epoch in range(epoches):
        for bs, _ in dataloader:
            x = bs['values'].to(torch.float).to(device)
            masks = bs['masks'].to(torch.float).to(device)
            deltas = bs['deltas'].to(torch.float).to(device)
            generated_ts = model.generator(x, masks, deltas)
            D_fake = model.discriminator(generated_ts, masks, deltas)
            D_f_loss = torch.mean(D_fake)

            D_real = model.discriminator(x, masks, deltas)
            D_r_loss = torch.mean(D_real)

            loss = D_f_loss - D_r_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for p in model.discriminator.parameters():
                p.data.clamp_(-0.1, 0.1)
            
            D_r_losses.append(D_r_loss.item())
            D_f_losses.append(D_f_loss.item())

    return np.mean(D_r_losses), np.mean(D_f_losses)

@torch.no_grad()
def evaluation(model, test_dataloader):
    losses= 0.0
    num_maskes = 0.0
    for bs, _ in test_dataloader:
        x = bs['values'].to(torch.float).to(device)
        masks = bs['masks'].to(torch.float).to(device)
        deltas = bs['deltas'].to(torch.float).to(device)
        generated_ts = model.generator(x, masks, deltas)
        losses += torch.sum((x - generated_ts)**2 * (1 - masks)).item()
        num_maskes += torch.sum(1 - masks).sum()
    return losses / num_maskes

def train(model, train_dataloader, test_dataloader, logs_dir, model_save_dir):
    writer = SummaryWriter(logs_dir)

    optim_G = torch.optim.Adam(model.generator.parameters(), lr=0.005)
    optim_D = torch.optim.Adam(model.discriminator.parameters(), lr = 0.005)
    ## first pretrain
    pretrain(model, train_dataloader, optim_G)
    
    best_val_loss = float('inf')
    for epoch in tqdm(range(200)):
        
        model.train()
        mse_loss, D_f_loss = optim_generator(model, train_dataloader, optim_G)
        D_r_loss, D_f_loss = optim_discriminator(model, train_dataloader, optim_D)
        
        model.eval()
        val_loss = evaluation(model, test_dataloader)
        
        writer.add_scalar('loss/D_f_loss', D_f_loss, epoch)
        writer.add_scalar('loss/D_r_loss', D_r_loss, epoch)
        writer.add_scalar('loss/mse_loss', mse_loss, epoch)
        writer.add_scalar('loss/val_loss', val_loss, epoch)

        if epoch % 1 == 0:
            print('Epoach {}, Training....., D_r_loss:{}'.format(epoch, D_r_loss))
            print('Epoach {}, Training....., D_f_loss:{}'.format(epoch, D_f_loss))
            print('Epoach {}, Training....., mse_loss:{}'.format(epoch, mse_loss))
            print('Epoach {}, Training....., val_loss:{}'.format(epoch, val_loss))
        
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            torch.save(model.state_dict(), f'{model_save_dir}best_model.pth')
                    
if __name__ == '__main__':

    UEASummary = pd.read_csv('./datasets/Data/MultivariateUEA/Summary.csv')
    problem_index_list = [17,12, 25, 27]

    for problem_ind in range(len(problem_index_list)):
        i = problem_index_list[problem_ind]
        problem = UEASummary.iloc[i]['Problem']
        sequence_length = UEASummary.iloc[i]['SeriesLength']
        input_dim = UEASummary.iloc[i]['NumDimensions']
        MTS_size = [UEASummary.iloc[i]['SeriesLength'], UEASummary.iloc[i]['NumDimensions']]
        train_dataset = MultivariateUEA_train(problem, MTS_size)
        test_dataset = MultivariateUEA_test(problem, MTS_size)
        
        model = E2EGanImputation(input_dim=input_dim, hidden_dim=64).to(device)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_dataloadr = DataLoader(test_dataset, batch_size=64, shuffle=False)

        logs_dir = f'logs/E2EGanImputation/{problem}/'
        model_save_dir = f'TrainedModel/E2EGanImputation/{problem}/'
        train(model, train_dataloader, test_dataloadr, logs_dir, model_save_dir)