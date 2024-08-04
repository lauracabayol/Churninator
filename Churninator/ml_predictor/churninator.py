import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import torch
import json
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

#import sys
#sys.path.append("../utils")
from Churninator.utils.optuna_optimizers import optimize_gb_with_optuna, optimize_nn_with_optuna
from Churninator.utils.plots import plot_feature_importances_seaborn

from Churninator.data_processing.data_cleaning import data_cleaner
from Churninator.ml_predictor.network_architecture import ChurnNet

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
os.environ['PYTHONHASHSEED'] = '0'

# Ensure deterministic behavior
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled=False 

class Churninator():
    def __init__(self,
                path_to_file,
                features2use,
                preprocess_file=False,
                algorithm_bestparams=None,
                optimize_optuna=False,
                verbose=False,
                oversample=True, 
                undersample=False,
                algorithm='GB'
                ):


        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.algorithm=algorithm
        self.oversample=oversample
        self.undersample=undersample
        
        if preprocess_file:
            print("Preprossing file, this can take ~5', but it only needs to be done once")
            DataCleaner = data_cleaner(path_to_data=path_to_file,
                                       labels_to_encode = ['Gender', 'Country'],
                                       save_file_path = '../data/cleant_data.csv',
                                       verbose=True,
                                       make_plots=False)
            
        else:
            df = pd.read_csv(path_to_file, header=0, sep=',')
                        
        X,y = df[features2use], df['Exited']


        if self.algorithm == 'NN':
            # define model
            
            #normalize data
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        self._set_training_data(X,y)
        
        if algorithm_bestparams is not None:
            if verbose:
                print("Loading optimized parameters form file")
            f = open(algorithm_bestparams) 
            best_params = json.load(f) 
        else:
            if optimize_optuna is False:
                raise AttributeError("Algorithm parameters not provided and optimize_optuna set to False. Please provide parameters or enable optimize_optuna.")
            else:             
                if self.algorithm=='GB':
                    if verbose:
                        print("Optimizing GB parameters with optuna")
                    best_params = optimize_gb_with_optuna(self.X_train.values, self.y_train.values.flatten(), n_trials=20)
                    with open('../data/bestparams_GB.json', 'w') as fp:
                        json.dump(best_params, fp) 
                elif self.algorithm=='NN':
                    if verbose:
                        print("Optimizing NN parameters with optuna")

                    best_params = optimize_nn_with_optuna(self._train_nn, 
                                                          n_trials=40)
                    
                    with open('../data/bestparams_NN.json', 'w') as fp:
                        json.dump(best_params, fp) 
                else:
                    raise AttributeError("Accepted algorithm options are 'GB' and 'NN'")

        if self.algorithm == 'GB':
            self.gb_optimized = GradientBoostingClassifier(**best_params, random_state=42)
            self.gb_optimized.fit(self.X_train.values, self.y_train.values.flatten())

        elif self.algorithm=='NN':                    
            self.nn_optimized= ChurnNet(input_dim = len(features2use), 
                                         hidden_dim= best_params['hidden_dim'], 
                                         output_dim=1, 
                                         nhidden = best_params['nhidden'])

            model, _ = self._train_nn(model=self.nn_optimized,
                          learning_rate=best_params['learning_rate'], 
                          batch_size=best_params['batch_size'],
                          optimizer_name=best_params['optimizer'], 
                          num_epochs=best_params['num_epochs'])

    def _set_training_data(self, X, y):

                    
        self.X_train, X_val, self.y_train, y_val = train_test_split(X, 
                                                          y, 
                                                          test_size=0.2, 
                                                          random_state=42, 
                                                          stratify=y)
        
        self.X_test, self.X_val, self.y_test, self.y_val = train_test_split(X_val, 
                                                        y_val, 
                                                        test_size=0.5, 
                                                        random_state=42, 
                                                        stratify=y_val)

        if self.oversample and self.undersample:
            raise AttributeError("Select either oversample or undersample")
        elif self.oversample:
            if self.verbose:
                print("Oversampling training data")
            oversampler = RandomOverSampler(random_state=42)
            self.X_train, self.y_train = oversampler.fit_resample(self.X_train, self.y_train)
        elif self.undersample:
            if self.verbose:
                print("Undersampling training data")
            undersampler = RandomUnderSampler(random_state=42)
            self.X_train, self.y_train = undersampler.fit_resample(self.X_train, self.y_train)



    def get_validation_data(self):
        return self.X_val, self.y_val
        
    def get_test_data(self):
        return self.X_test, self.y_test

    def get_training_data(self):
        return self.X_train, self.y_train

    def predict_GB(self, X_test):
        y_pred = self.gb_optimized.predict_proba(X_test)[:, 1]
        return y_pred

    def predict_NN(self, X_test):
        self.nn_optimized =  self.nn_optimized.eval()
        with torch.no_grad():
            y_pred = self.nn_optimized(torch.Tensor(X_test).to(self.device)).detach().cpu().numpy()
        return y_pred
    
    def plot_ROC(self, X_test, y_test):
        if self.algorithm=='GB':
            y_pred = self.gb_optimized.predict_proba(X_test)[:, 1]
        elif self.algorithm=='NN':
            y_pred=self.predict_NN(X_test)           
        RocCurveDisplay.from_predictions(y_test,y_pred)

    def compute_confusion_matrix(self,X_test, y_test, optimize_threshold=True, plot=True):
        if self.algorithm=='GB':
            y_pred = self.gb_optimized.predict_proba(X_test)[:, 1]
        elif self.algorithm=='NN':
            y_pred=self.predict_NN(X_test)
            
        if not optimize_threshold:
            print("Setting cut probability threshold to 0.5")
        else:
            threshold_cut = self._select_threshold_4binary(y_pred,y_test)
            
        ypred_categorical = (y_pred >= threshold_cut).astype(int)
        cf = confusion_matrix(y_test.values.flatten(), ypred_categorical,normalize='true')
        
        if plot:
            disp = ConfusionMatrixDisplay(cf)
            disp.plot()
            plt.show()
            
            return cf
        else:
            return cf
            

    def _select_threshold_4binary(self,y_pred,y_true):
        precision, recall, thresholds = precision_recall_curve(y_true.values.flatten(), y_pred)
        fscore = (2 * precision * recall) / (precision + recall)
        
        # locate the index of the largest f score
        ix = np.argmax(fscore)
        if self.verbose:
            print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
        return thresholds[ix]

    
    def _set_dataloaders(self, batch_size):

        g = torch.Generator()
        g.manual_seed(0)

        train_dataset = TensorDataset(torch.tensor(self.X_train, dtype=torch.float32), 
                                      torch.tensor(self.y_train.values.flatten(), dtype=torch.float32).unsqueeze(1))
        val_dataset = TensorDataset(torch.tensor(self.X_val, dtype=torch.float32),
                                    torch.tensor(self.y_val.values.flatten(), dtype=torch.float32).unsqueeze(1))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, generator=g)
        dataloaders = {'train': train_loader, 'val': val_loader}
        return dataloaders


    def _train_nn(self, model, batch_size, learning_rate, optimizer_name, num_epochs):
    
        model = model.to(self.device)
        
        # Use weighted BCE Loss
        criterion = nn.BCELoss()
    
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'wAdam':
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'Adamax':
            optimizer = optim.Adamax(model.parameters(), lr=learning_rate)

        dataloaders = self._set_dataloaders(batch_size)
        
        for epoch in range(num_epochs):
            if self.verbose:
                print(f'Epoch {epoch}/{num_epochs-1}')
                print('-' * 10)
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
    
                running_loss = 0.0
    
                # Iterate over data
                for inputs, labels in dataloaders[phase]:
                    
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device).squeeze(1)
                    
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs.squeeze(), labels)
                        #loss = custom_loss_with_bias_regularization(outputs, labels, model, lambda_b)
                        preds = (outputs > 0.5).float()
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    #scheduler.step()
    
                    running_loss += loss.item() * inputs.size(0)
                
                epoch_loss = running_loss / len(dataloaders[phase].dataset)

                if self.verbose:
                    print(f'{phase} Loss: {epoch_loss:.4f}')
    
        return model, dataloaders
    
    
