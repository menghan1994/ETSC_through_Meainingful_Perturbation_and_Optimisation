# Supplementary Materials Repository
Welcome to the Supplementary Materials Repository for our work **Explaining Time Series Classifiers trhough Meaningful Perturbation and Optimisation**. This repository contains additional resources that support our research findings.





###
To run the full experiment. Download the UEA datasets and put it into the folder /dataset/Data/MultivariateUEA/... you need to process the original datasets and save it in a numpy file. 


Train_Classifier.py -- Train the classier to be explained. 
Train_GenModel.py -- Train the generative model for realistic inputs generation.
main_train_E2Gan.py and main_rain_BRITS.py train two time series imputation models. 

Explaining.py --- explain Trained models. 
Evaluation.py -- evaluation the results. 
