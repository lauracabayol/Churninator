"""
ChurnNet Module
===============

This module contains the ChurnNet class for churn prediction.
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
import matplotlib 
matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12)
import torch
import json
import os
import subprocess


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

from Churninator.utils.optuna_optimizers import optimize_gb_with_optuna, optimize_nn_with_optuna
from Churninator.utils.plots import plot_feature_importances_seaborn

from Churninator.data_processing.data_cleaning import DataCleaner
from Churninator.ml_predictor.network_architecture import ChurnNet

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
os.environ['PYTHONHASHSEED'] = '0'

# Ensure deterministic behavior
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

class Churninator:
    """
    A class to handle customer churn prediction using either Gradient Boosting or a Neural Network.

    Attributes:
        device (torch.device): Device to run the model on (CPU or GPU).
        verbose (bool): Verbose flag for printing progress information.
        algorithm (str): The algorithm to use ('GB' for Gradient Boosting, 'NN' for Neural Network).
        oversample (bool): Flag to determine whether to oversample the training data.
        undersample (bool): Flag to determine whether to undersample the training data.

    Methods:
        get_validation_data: Returns the validation data.
        get_test_data: Returns the test data.
        get_training_data: Returns the training data.
        predict_GB: Predicts probabilities using the Gradient Boosting model.
        predict_NN: Predicts probabilities using the Neural Network model.
        plot_ROC: Plots the ROC curve.
        compute_confusion_matrix: Computes and optionally plots the confusion matrix.
    """

    def __init__(self, path_to_file, features2use, preprocess_file=False, algorithm_bestparams=None,
                 optimize_optuna=False, verbose=False, oversample=True, undersample=False, algorithm='GB'):
        """
        Initializes the Churninator class.

        Args:
            path_to_file (str): Path to the dataset file.
            features2use (list): List of features to use for the model.
            preprocess_file (bool): Whether to preprocess the file.
            algorithm_bestparams (str): Path to the file containing optimized parameters.
            optimize_optuna (bool): Whether to optimize parameters using Optuna.
            verbose (bool): Verbosity flag.
            oversample (bool): Whether to oversample the training data.
            undersample (bool): Whether to undersample the training data.
            algorithm (str): Algorithm to use ('GB' for Gradient Boosting, 'NN' for Neural Network).
        """
        self.root_repo=subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.algorithm = algorithm
        self.oversample = oversample
        self.undersample = undersample

        if preprocess_file:
            print("Preprocessing file, this can take ~5', but it only needs to be done once")
            data_cleaner = DataCleaner(path_to_data=path_to_file,
                                       labels_to_encode=['Gender', 'Country'],
                                       save_file_path=self.root_repo+'/data/clean_data.csv',
                                       verbose=True,
                                       make_plots=False)
        else:
            df = pd.read_csv(path_to_file, header=0, sep=',')

        X, y = df[features2use], df['Exited']

        if self.algorithm == 'NN':
            # Normalize data
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        self._set_training_data(X, y)

        if algorithm_bestparams is not None:
            if verbose:
                print("Loading optimized parameters from file")
            with open(algorithm_bestparams) as f:
                best_params = json.load(f)
        else:
            if not optimize_optuna:
                raise AttributeError("Algorithm parameters not provided and optimize_optuna set to False. "
                                     "Please provide parameters or enable optimize_optuna.")
            else:
                if self.algorithm == 'GB':
                    if verbose:
                        print("Optimizing GB parameters with Optuna")
                    best_params = optimize_gb_with_optuna(self.X_train.values, self.y_train.values.flatten(), n_trials=20)
                    with open(self.root_repo+'/data/bestparams_GB.json', 'w') as fp:
                        json.dump(best_params, fp)
                elif self.algorithm == 'NN':
                    if verbose:
                        print("Optimizing NN parameters with Optuna")
                    best_params = optimize_nn_with_optuna(self._train_nn, n_trials=40)
                    with open(self.root_repo+'/data/bestparams_NN.json', 'w') as fp:
                        json.dump(best_params, fp)
                else:
                    raise AttributeError("Accepted algorithm options are 'GB' and 'NN'")

        if self.algorithm == 'GB':
            self.gb_optimized = GradientBoostingClassifier(**best_params, random_state=42)
            self.gb_optimized.fit(self.X_train.values, self.y_train.values.flatten())
        elif self.algorithm == 'NN':
            self.nn_optimized = ChurnNet(input_dim=len(features2use),
                                         hidden_dim=best_params['hidden_dim'],
                                         output_dim=1,
                                         nhidden=best_params['nhidden'])
            self._train_nn(model=self.nn_optimized,
                           learning_rate=best_params['learning_rate'],
                           batch_size=best_params['batch_size'],
                           optimizer_name=best_params['optimizer'],
                           num_epochs=best_params['num_epochs'])

    def _set_training_data(self, X, y):
        """
        Splits data into training, validation, and test sets. Optionally oversamples or undersamples the training data.

        Args:
            X (pd.DataFrame or np.ndarray): Feature data.
            y (pd.Series or np.ndarray): Target labels.
        """
        self.X_train, X_val, self.y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.X_test, self.X_val, self.y_test, self.y_val = train_test_split(X_val, y_val, test_size=0.5, random_state=42, stratify=y_val)

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
        """Returns the validation data."""
        return self.X_val, self.y_val

    def get_test_data(self):
        """Returns the test data."""
        return self.X_test, self.y_test

    def get_training_data(self):
        """Returns the training data."""
        return self.X_train, self.y_train

    def predict_GB(self, X_test):
        """
        Predicts probabilities using the Gradient Boosting model.

        Args:
            X_test (pd.DataFrame or np.ndarray): Test feature data.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        return self.gb_optimized.predict_proba(X_test)[:, 1]

    def predict_NN(self, X_test):
        """
        Predicts probabilities using the Neural Network model.

        Args:
            X_test (pd.DataFrame or np.ndarray): Test feature data.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        self.nn_optimized.eval()
        with torch.no_grad():
            y_pred = self.nn_optimized(torch.Tensor(X_test).to(self.device)).detach().cpu().numpy()
        return y_pred

    def plot_ROC(self, X_test, y_test):
        """
        Plots the ROC curve.

        Args:
            X_test (pd.DataFrame or np.ndarray): Test feature data.
            y_test (pd.Series or np.ndarray): True labels.
        """
        if self.algorithm == 'GB':
            y_pred = self.gb_optimized.predict_proba(X_test)[:, 1]
        elif self.algorithm == 'NN':
            y_pred = self.predict_NN(X_test)
        RocCurveDisplay.from_predictions(y_test, y_pred)

    def compute_confusion_matrix(self, X_test, y_test, optimize_threshold=True, plot=True):
        """
        Computes and optionally plots the confusion matrix.

        Args:
            X_test (pd.DataFrame or np.ndarray): Test feature data.
            y_test (pd.Series or np.ndarray): True labels.
            optimize_threshold (bool): Whether to optimize the threshold.
            plot (bool): Whether to plot the confusion matrix.

        Returns:
            np.ndarray: Confusion matrix.
        """
        if self.algorithm == 'GB':
            y_pred = self.gb_optimized.predict_proba(X_test)[:, 1]
        elif self.algorithm == 'NN':
            y_pred = self.predict_NN(X_test)

        if not optimize_threshold:
            print("Setting cut probability threshold to 0.5")
            threshold_cut = 0.5
        else:
            threshold_cut = self._select_threshold_4binary(y_pred, y_test)

        y_pred_categorical = (y_pred >= threshold_cut).astype(int)
        cf = confusion_matrix(y_test.values.flatten(), y_pred_categorical, normalize='true')

        if plot:
            disp= ConfusionMatrixDisplay.from_predictions(y_test.values.flatten(), 
                                                          y_pred_categorical,
                                                          display_labels=['Non Exited', 'Exited'],
                                                          colorbar=False,
                                                          normalize='true')
            #disp = ConfusionMatrixDisplay(cf, display_labels=['Non Exited', 'Exited'], colorbar=True)
            #disp.plot()
            plt.show()

        return cf

    def _select_threshold_4binary(self, y_pred, y_true):
        """
        Selects the optimal threshold for binary classification based on F-score.

        Args:
            y_pred (np.ndarray): Predicted probabilities.
            y_true (pd.Series or np.ndarray): True labels.

        Returns:
            float: Optimal threshold.
        """
        precision, recall, thresholds = precision_recall_curve(y_true.values.flatten(), y_pred)
        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.argmax(fscore)
        if self.verbose:
            print(f'Best Threshold={thresholds[ix]}, F-Score={fscore[ix]:.3f}')
        return thresholds[ix]

    def _set_dataloaders(self, batch_size):
        """
        Creates DataLoader objects for training and validation datasets.

        Args:
            batch_size (int): Size of the batches.

        Returns:
            dict: Dictionary containing training and validation DataLoader objects.
        """
        g = torch.Generator()
        g.manual_seed(0)

        train_dataset = TensorDataset(torch.tensor(self.X_train, dtype=torch.float32),
                                      torch.tensor(self.y_train.values.flatten(), dtype=torch.float32).unsqueeze(1))
        val_dataset = TensorDataset(torch.tensor(self.X_val, dtype=torch.float32),
                                    torch.tensor(self.y_val.values.flatten(), dtype=torch.float32).unsqueeze(1))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, generator=g)
        return {'train': train_loader, 'val': val_loader}

    def _train_nn(self, model, batch_size, learning_rate, optimizer_name, num_epochs):
        """
        Trains the Neural Network model.

        Args:
            model (nn.Module): Neural Network model to train.
            batch_size (int): Size of the batches.
            learning_rate (float): Learning rate for the optimizer.
            optimizer_name (str): Name of the optimizer to use.
            num_epochs (int): Number of epochs to train the model.

        Returns:
            nn.Module: Trained model.
            dict: Dictionary containing training and validation DataLoader objects.
        """
        model = model.to(self.device)
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
                print(f'Epoch {epoch}/{num_epochs - 1}')
                print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device).squeeze(1)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs.squeeze(), labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)

                if self.verbose:
                    print(f'{phase} Loss: {epoch_loss:.4f}')

        return model, dataloaders
