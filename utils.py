
from models.predmodel import TSPredModel
from models.DistributionApproximation.TimeSeriesGen import TimeSeriesGen_GAN
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from models.TransformerImputation.TransformerImputation import DetermineTransformer, DetermineTransformer2

from datasets.MultivariateUEA import MultivariateUEA_test
from datasets.mnist import MNIST_test
from models.TemporalConvNet import TCN



def load_classifier_and_generativemodel(ind, model_type, gentype = 'Transformer'):
    if ind != None:
        UEASummary = pd.read_csv('./datasets/Data/MultivariateUEA/Summary.csv')
        problem = UEASummary.iloc[ind]['Problem']
        test_dataset = MultivariateUEA_test(problem)
        num_classes = UEASummary.iloc[ind]['NumClasses']
        input_dim = UEASummary.iloc[ind]['NumDimensions']
        sequence_length = UEASummary.iloc[ind]['SeriesLength']

        model_label = problem
        dataset = 'MultivariateUEA'
        #  load Classifier ....
        if model_type == 'LSTM':
            latent_dim = 64
            
            modelsave_params = {
                'model_save_dir': f'TrainedModel/{dataset}/PredModel/{model_label}/'
            }
            net = TSPredModel(input_dim=input_dim, num_classes = num_classes, hidden_size=latent_dim, train_dataset=None, test_dataset=test_dataset, val_dataset=test_dataset)
            model_save_name = os.path.join(modelsave_params['model_save_dir'], 'best.pth')
            net.load_state_dict(torch.load(model_save_name))
            net.eval()
        elif model_type == 'TCN':
            net = TCN(input_size = input_dim, 
                                    output_size = num_classes,
                                    num_channels = [25] * 8, 
                                    kernel_size = 7, 
                                    dropout = 0.2, 
                                    train_dataset = None, 
                                    test_dataset = test_dataset)
            modelsave_params = {
            'model_save_dir': f'TrainedModel/{dataset}/{model_type}/{model_label}/'
            }
            model_save_name = os.path.join(modelsave_params['model_save_dir'], 'best.pth')
            net.load_state_dict(torch.load(model_save_name, map_location=torch.device('cpu')))
            net.eval()
            

        if gentype == 'Transformer':
            params = {
                'input_dim': input_dim, 
                'lr' :0.001
            }        
            model_save_path = f'TrainedModel/TransformerImputation/{problem}/best_model.ckpt'
            GenModel = DetermineTransformer.load_from_checkpoint(checkpoint_path = model_save_path, params = params, train_dataset = None, test_dataset = test_dataset)
        
        elif gentype == 'RNN':

            latent_dim = 128
            kl_lambda = 0.0001
            rec_lambda = 1.0

            dataset = 'MultivariateUEA'
            model_label = problem
            GenModel = TimeSeriesRNNVAEGen_GAN(latent_dim, kl_lambda, rec_lambda, input_dim = input_dim, sequence_length=sequence_length, train_dataset=None, test_dataset=None, val_dataset=None)
            
            genmodelsave_params = {
                'model_save_dir': f'TrainedModel/GEN/{dataset}/{model_label}'
            }
            genmodel_save_name = os.path.join(genmodelsave_params['model_save_dir'], 'best.pth')
            GenModel.load_state_dict(torch.load(genmodel_save_name, map_location=torch.device('cpu')))
            GenModel = GenModel.generator
    else:

        test_dataset = MNIST_test()
        
        params = {
                'input_dim': 28, 
                'lr' :0.001
            }        
        model_save_path = f'TrainedModel/TransformerImputation/MNIST/best_model.ckpt'
        GenModel = DetermineTransformer2.load_from_checkpoint(checkpoint_path = model_save_path, params = params, train_dataset = None, test_dataset = test_dataset)
        input_dim = 28
        num_classes = 10
        dataset = 'MNISTClassifier'
        if model_type == 'LSTM':
            latent_dim = 64
            
            model_save_dir = f'TrainedModel/{dataset}/{model_type}/best_model.ckpt'
            net = TSPredModel.load_from_checkpoint(checkpoint_path=model_save_dir, input_dim=input_dim, num_classes = num_classes, hidden_size=latent_dim, train_dataset=None, test_dataset=test_dataset, val_dataset=test_dataset)

            net.eval()


        elif model_type == 'TCN':
            model_save_dir = f'TrainedModel/{dataset}/{model_type}/best_model.ckpt'
            net = TCN.load_from_checkpoint(checkpoint_path=model_save_dir, 
                                            input_size = input_dim, 
                                    output_size = num_classes,
                                    num_channels = [25] * 8, 
                                    kernel_size = 7, 
                                    dropout = 0.2, 
                                    train_dataset = None, 
                                    test_dataset = test_dataset)
            net.eval()



    return net, GenModel, test_dataset
