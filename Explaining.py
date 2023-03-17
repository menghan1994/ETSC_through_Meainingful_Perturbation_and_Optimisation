

import pandas as pd

import os
import torch
import numpy as np
from utils import load_classifier_and_generativemodel

import time
from tqdm import tqdm
from explainer.Saliency import saliency


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--indexes', default='0_1',type=str, required=False)
parser.add_argument('--device', default='cuda:0', type=str, required=False)
parser.add_argument('--num_screens', default='1', type=str, required=False)

args = parser.parse_args()
list_of_dataset = args.indexes.split('_')
index_start,  index_end = int(list_of_dataset[0]), int(list_of_dataset[1])
device = args.device
num_screens = int(args.num_screens)


def explainer(method, test_dataset, net, GenModel, problem, net_type):

    np.random.seed(0)
    sampled_ind = np.random.choice(np.arange(len(test_dataset)), min(1000, len(test_dataset)), replace = False)

    input_dim = test_dataset[0][0]['values'].shape[-1]
    sequence_len = test_dataset[0][0]['values'].shape[-2]

    for ind in tqdm(range(len(sampled_ind))):
        start_time = time.time()
        i = sampled_ind[ind]
        x, y = test_dataset[i][0]['values'], test_dataset[i][1]
        x = torch.from_numpy(x).unsqueeze(0)
        x = x.to(torch.float).to(device)

        target = net.predict(x)            
        attr, _ = saliency(net, x, target, method, GenModel = GenModel, init_scaler = (4, input_dim), device = device)
        Saliency = torch.from_numpy(attr).unsqueeze(0)
    
        total_time = time.time() - start_time
    
        save_dir = f'Saliency/{net_type}/{method}/{problem}/{ind}/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(Saliency, save_dir + 'saliency.pt')

        with open(f'{save_dir}time_expense.txt', 'a+') as f:
            f.write(problem)
            f.write(",")
            f.write(str(total_time/len(sampled_ind)))
            f.write('\n')
        
            # explaining_attrs_and_input(x, attr, net, GenModel, save_dir, 'explanations.png')

def main():
    UEASummary = pd.read_csv('./datasets/Data/MultivariateUEA/Summary.csv')
    for net_type in ['LSTM']:
        for ind in [0, 2, 3, 7, 9, 15, 16, 17, 18, 20, 21, 23, 24, 27, None, 12, 25, 29, 28, 4]:
        # for ind in [25, 29, 28]:
            for method in ['IntegratedGradients', 'Saliency', 'LIME','OurSearch_BPSO', 'OurSearch_GA_parallel']:
                try:
                    net, GenModel, test_dataset = load_classifier_and_generativemodel(ind, net_type, 'Transformer')
                except:
                    net, GenModel, test_dataset = load_classifier_and_generativemodel(ind, net_type, 'RNN')

                net.to(device)
                GenModel.to(device)
                
                problem = 'MNIST' if ind == None else UEASummary.iloc[ind]['Problem']
                explainer(method, test_dataset, net, GenModel, problem, net_type)

if __name__ == '__main__':
    main()