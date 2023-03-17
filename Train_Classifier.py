from re import I
from Trainer.Trainer import Trainer
from models.predmodel import TSPredModel
from datasets.MultivariateUEA import MultivariateUEA_train, MultivariateUEA_test
import pandas as pd
from datasets.mnist import MNIST_train, MNIST_test
from models.TemporalConvNet import TCN
from models.Transformer import Transformer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

def main_UEA():

    UEASummary = pd.read_csv('./datasets/Data/MultivariateUEA/Summary.csv')

    for i in range(len(UEASummary)):
        problem = UEASummary.iloc[i]['Problem']
        train_dataset = MultivariateUEA_train(problem)
        test_dataset = MultivariateUEA_test(problem)
        sequence_length = UEASummary.iloc[i]['SeriesLength']
        
        latent_dim = 64

        num_classes = UEASummary.iloc[i]['NumClasses']
        input_dim = UEASummary.iloc[i]['NumDimensions']

        if model_type == 'LSTM':
            model = TSPredModel(input_dim=input_dim, num_classes = num_classes, hidden_size=latent_dim, train_dataset=train_dataset, test_dataset=test_dataset, val_dataset=test_dataset)

        if model_type == 'TCN':
            model = TCN(input_size = input_dim, 
                                output_size = num_classes,
                                num_channels = [25] * 8, 
                                kernel_size = 7, 
                                dropout = 0.2, 
                                train_dataset = train_dataset, 
                                test_dataset = test_dataset)

        logger = TensorBoardLogger("logs", name=f"TrainedModel/problem/{model_type}/")

        checkpoint_callback = ModelCheckpoint(
            save_top_k= 1,
            monitor="val_loss",
            mode="min",
            dirpath=f"TrainedModel/problem/{model_type}/",
            filename="best_model",
        )
        trainer = pl.Trainer(accelerator="gpu", devices=[1], max_epochs=200, logger = logger, callbacks=[checkpoint_callback])
        trainer.fit(model)

def main_train_MNIST_classifier(model_type):

    train_dataset, test_dataset = MNIST_train(), MNIST_test()

    input_dim = 28
    num_classes = 10
    latent_dim = 64
    if model_type == 'LSTM':
        model = TSPredModel(input_dim=input_dim, num_classes = num_classes, hidden_size=latent_dim, train_dataset=train_dataset, test_dataset=test_dataset, val_dataset=test_dataset)

    if model_type == 'TCN':
        model = TCN(input_size = input_dim, 
                            output_size = num_classes,
                            num_channels = [25] * 8, 
                            kernel_size = 7,
                            dropout = 0.2, 
                            train_dataset = train_dataset, 
                            test_dataset = test_dataset)
    logger = TensorBoardLogger("logs", name=f"TrainedModel/MNISTClassifier/{model_type}/")

    checkpoint_callback = ModelCheckpoint(
        save_top_k= 3,
        monitor="val_loss",
        mode="min",
        dirpath=f"TrainedModel/MNISTClassifier/{model_type}/",
        filename="best_model",
    )
    trainer = pl.Trainer(max_epochs=20, logger = logger, callbacks=[checkpoint_callback])
    trainer.fit(model)

if __name__ == '__main__':
    for model_type in ['LSTM']:
        main_train_MNIST_classifier(model_type)
