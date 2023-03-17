from re import I
from Trainer.Trainer import Trainer
from models.predmodel import TSPredModel
from datasets.MultivariateUEA import MultivariateUEA_train, MultivariateUEA_test
import pandas as pd
from models.BRITS.BRITS import BIRTS
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import matplotlib.pyplot as plt
from models.TransformerImputation.TransformerImputation import DetermineTransformer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--indexes', default='0_1',type=str, required=False)
parser.add_argument('--device', default='cuda:1', type=str, required=False)
parser.add_argument('--num_screens', default='1', type=str, required=False)

args = parser.parse_args()
list_of_dataset = args.indexes.split('_')
index_start = int(list_of_dataset[0])
device = args.device
num_screens = int(args.num_screens)


def main(withmissing = True):

    UEASummary = pd.read_csv('./datasets/Data/MultivariateUEA/Summary.csv')
    problem_index_list = [9, 12, 17, 27, 29, 25]

    for problem_ind in range(index_start, len(problem_index_list), num_screens):
        i = problem_index_list[problem_ind]
        problem = UEASummary.iloc[i]['Problem']
        sequence_length = UEASummary.iloc[i]['SeriesLength']
        MTS_size = [UEASummary.iloc[i]['SeriesLength'], UEASummary.iloc[i]['NumDimensions']]
        
        train_dataset = MultivariateUEA_train(problem, MTS_size)
        test_dataset = MultivariateUEA_test(problem, MTS_size)

        num_classes = UEASummary.iloc[i]['NumClasses']
        input_dim = UEASummary.iloc[i]['NumDimensions']

        birts = BIRTS(input_dim, 32, None, None, device)
        birts = birts.to(device)

        optimizer = optim.Adam(birts.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

        best_val_loss = float('inf')
        num_epochs = 1000
        for epoch in range(num_epochs):
            train_loss = train(birts, train_dataset, optimizer)
            val_loss = evaluation(birts, test_dataset)
            scheduler.step(val_loss) # Update the learning rate based on the validation loss
            print(f"Epoch {epoch+1} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}")
            
            model_save_dir = f'TrainedModel/BIRTS/{problem}/'
   
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                torch.save(birts.state_dict(), f'{model_save_dir}best_model.pt')
        
            # Log the training and validation loss history
            with open(f'{model_save_dir}loss_history.txt', 'a') as f:
                f.write(f"Epoch {epoch+1} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}\n")


def train(model, dataset, optimizer):
    model.train()

    running_loss = 0.0
    for bs in tqdm(DataLoader(dataset, batch_size=64)):
        data = {
            'forward':bs[0],
            'backward': bs[0],
            'labels':   bs[1],
            }
        loss = model(data)['loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataset)

@torch.no_grad()
def evaluation(model, dataset):
    model.eval()

    running_loss = 0.0
    for bs in DataLoader(dataset, batch_size=64):
        data = {
            'forward':bs[0],
            'backward': bs[0],
            'labels':   bs[1],
            }
        loss = model(data)['loss']

        running_loss += loss.item()
    return running_loss / len(dataset)



if __name__ == '__main__':
    main()
