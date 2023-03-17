
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
device = 'cuda:1'
from utils import load_classifier_and_generativemodel
import json
import os
@torch.no_grad()
def main():
    UEASummary = pd.read_csv('./datasets/Data/MultivariateUEA/Summary.csv', index_col=0)
    evaluation_results = {}
    
    for i in [0, 2, 3, 7, 9, 15, 16, 17, 18, 20, 21, 23, 24, 27, None, 12, 25, 29, 28]:
        for method in ['IntegratedGradients', 'Saliency', 'LIME','OurSearch_BPSO', 'OurSearch_GA_parallel'][1:]:
            problem = UEASummary.iloc[i]['Problem'] if i != None else 'MNIST'
            evaluation_results[problem] ={}
            evaluation_results[problem][method] ={}
            
            print(problem)

            try:
                net, GenModel, test_dataset = load_classifier_and_generativemodel(i, 'LSTM', 'Transformer')
            except:
                net, GenModel, test_dataset = load_classifier_and_generativemodel(i, 'LSTM', 'RNN')
                
            net.to(device)
            GenModel.to(device)

            np.random.seed(0)
            sampled_ind = np.random.choice(np.arange(len(test_dataset)), min(20, len(test_dataset)), replace = False)
            
            sparsity_scores = []
            for ind in tqdm(range(len(sampled_ind))):
                sample_ind = sampled_ind[ind]
                x = test_dataset[sample_ind][0]['values']
                x = torch.from_numpy(x).unsqueeze(0)
                x = x.to(torch.float).to(device)
                target = net.predict(x).item()
                save_dir = f'Saliency/TCN/{method}/{problem}/{ind}/saliency.pt'
                saliency = torch.load(save_dir).squeeze(0)
                sorted_index = torch.argsort(saliency.reshape(-1), descending=True)

                for delt_index in tqdm(range(1, len(sorted_index) + 1)):
                    index_to_zero = sorted_index[:delt_index]
                    mask = torch.ones_like(saliency.reshape(-1))
                    mask[index_to_zero] = 0
                    mask = mask.reshape(saliency.shape)
                    mask = mask.to(device)
                    x_o = x * mask
                    x_o = x_o.to(torch.float)
                    mask = mask.to(torch.float)

                    preds = []
                    for _ in range(1):
                        perturbed_samples = GenModel.sample(x_o, mask)
                        pred = net.prob(perturbed_samples)
                        preds.append(pred)
                    preds = torch.cat(preds, dim = 0)

                    current_label = torch.argmax(preds.mean(0)).item()
                    
                    if current_label != target:
                        sparsity_scores.append(delt_index)
                        break
            results = {}
            results['sparsity_scores'] = sparsity_scores
            with open(f'Saliency/TCN/{method}/{problem}/evaluation.json', 'w') as f:
                json.dump(results, f)
        

if __name__ == '__main__':
    main()
