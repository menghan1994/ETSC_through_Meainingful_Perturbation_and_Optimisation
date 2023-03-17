from models.DistributionApproximation.TimeSeriesGen import TimeSeriesGen_GAN
from Trainer.Trainer_TimeSeriesGen import TimeSeriesGEN_Trainer
from datasets.MultivariateUEA import *
import pandas as pd
from datasets.mnist import MNIST_train, MNIST_test

def UAEmain():
    
    UEASummary = pd.read_csv('./datasets/Data/MultivariateUEA/Summary.csv')
    
    for i in [0, 2, 3, 7, 9, 15, 16, 17, 18, 20, 21, 23, 24, 27, 12, 25, 29, 28, 4]:
        problem = UEASummary.iloc[i]['Problem']
        sequence_length = UEASummary.iloc[i]['SeriesLength']
        MTS_size = [UEASummary.iloc[i]['SeriesLength'], UEASummary.iloc[i]['NumDimensions']]
        train_dataset = MultivariateUEA_train(problem, MTS_size)
        test_dataset = MultivariateUEA_test(problem, MTS_size)
        
        num_classes = UEASummary.iloc[i]['NumClasses']
        input_dim = UEASummary.iloc[i]['NumDimensions']

        dataset = problem
        params = {
            'input_dim': input_dim,
            'latent_dim': 32,
            'sequence_length': sequence_length,
            'lr':0.001
        }

        model = TimeSeriesGen_GAN(params = params, train_dataset=train_dataset, test_dataset=test_dataset, val_dataset=test_dataset)

        optim_params = {
                    'lr' : 0.0001,
                    'max_epoches' : 1000,
                }

        logs_params = {
            'log_dir' : f'logs/{dataset}/{dataset}'
        }

        modelsave_params = {
            'model_save_dir': f'TrainedModel/Transformer_Gen/{dataset}/'
        }

        trainer = TimeSeriesGEN_Trainer(optim_params=optim_params, logs_params=logs_params, modelsave_params=modelsave_params)
        trainer.fit(model)
    

def MNIST_main():
    
    
    train_dataset, test_dataset = MNIST_train(), MNIST_test()
      
    dataset = 'MNIST'
    params = {
        'input_dim': 28,
        'latent_dim': 32,
        'sequence_length': 28,
        'lr':0.001
    }

    model = TimeSeriesGen_GAN(params = params, train_dataset=train_dataset, test_dataset=test_dataset, val_dataset=test_dataset)

    optim_params = {
                'lr' : 0.0001,
                'max_epoches' : 1000,
            }

    logs_params = {
        'log_dir' : f'logs/{dataset}/{dataset}'
    }

    modelsave_params = {
        'model_save_dir': f'TrainedModel/Transformer_Gen/{dataset}/'
    }

    trainer = TimeSeriesGEN_Trainer(optim_params=optim_params, logs_params=logs_params, modelsave_params=modelsave_params)
    trainer.fit(model)


if __name__ == '__main__':
    MNIST_main()
