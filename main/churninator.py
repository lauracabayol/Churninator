import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import torch
import json


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import sys
sys.path.append("../utils")
from optuna_optimizers import optimize_gb_with_optuna
from plots import plot_feature_importances_seaborn

sys.path.append("./")
from data_cleaning import data_cleaner
from network_architecture import ChurnNet

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


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
            #normalize data
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        self._set_training_data(X,y)

        if algorithm_bestparams is not None:
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
                        json.dump(best_params_gb, fp)                    
                elif self.algorithm=='NN':
                    assert False
                else:
                    raise AttributeError("Accepted algorithm options are 'GB' and 'NN'")

        if self.algorithm == 'GB':
            self.gb_optimized = GradientBoostingClassifier(**best_params, random_state=42)
            self.gb_optimized.fit(self.X_train.values, self.y_train.values.flatten())


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

    def plot_ROC(self, X_test, y_test):
        y_pred = self.gb_optimized.predict_proba(X_test)[:, 1]
        RocCurveDisplay.from_predictions(y_test,y_pred)

    def compute_confusion_matrix(self,X_test, y_test, optimize_threshold=True, plot=True):
        y_pred = self.gb_optimized.predict_proba(X_test)[:, 1]
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
