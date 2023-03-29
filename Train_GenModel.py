from models.DistributionApproximation.TimeSeriesRnnVAEReconstructor import TimeSeriesRNNVAEGen_GAN
from Trainer.Trainer_TimeSeriesRNNVAEReconstructor import TimeSeriesRNNVAEReconstructorTrainer
from datasets.copyMultivariateUEA import MultivariateUEA_train, MultivariateUEA_test
import pandas as pd

def UAEmain():
    
    UEASummary = pd.read_csv('./datasets/Data/MultivariateUEA/Summary.csv')
    
    for i in [15]:
        problem = UEASummary.iloc[i]['Problem']
        sequence_length = UEASummary.iloc[i]['SeriesLength']
        MTS_size = [UEASummary.iloc[i]['SeriesLength'], UEASummary.iloc[i]['NumDimensions']]
        train_dataset = MultivariateUEA_train(problem, MTS_size)
        test_dataset = MultivariateUEA_test(problem, MTS_size)
        
        num_classes = UEASummary.iloc[i]['NumClasses']
        input_dim = UEASummary.iloc[i]['NumDimensions']

        latent_dim = 128
        kl_lambda = 0.00001
        rec_lambda = 1.0

        dataset = 'MultivariateUEA'
        model_label = problem

        print(problem)
        model = TimeSeriesRNNVAEGen_GAN(latent_dim, kl_lambda, rec_lambda, input_dim = input_dim, sequence_length=sequence_length, train_dataset=train_dataset, test_dataset=test_dataset, val_dataset=test_dataset)

        optim_params = {
                    'lr' : 0.0001,
                    'max_epoches' : 1000,
                }

        logs_params = {
            'log_dir' : f'runs/GEN_tunning-200/{dataset}/{model_label}'
        }

        modelsave_params = {
            'model_save_dir': f'TrainedModel/GEN_tunning-200/{dataset}/{model_label}'
        }

        trainer = TimeSeriesRNNVAEReconstructorTrainer(optim_params=optim_params, logs_params=logs_params, modelsave_params=modelsave_params)
        trainer.fit(model)

if __name__ == '__main__':
    UAEmain()
