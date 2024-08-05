# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: ecovadis
#     language: python
#     name: ecovadis
# ---

# # TUTORIAL FOR CHURN PREDICTION WITH NEURAL NETWORKS

# %load_ext autoreload
# %autoreload 2

import Churninator
from Churninator.ml_predictor.churninator import Churninator

import subprocess
root_directory=subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()

# ## These are the features that we will use to make predictions

features2use = ['CreditScore', 'Age', 'Tenure', 'Balance (EUR)', 'NumberOfProducts', 'IsActiveMember','EstimatedSalary', 'encoded_Country','encoded_sentiments','score']

# ## Calling the predictor. 
#     - The current configuration assumes the original file has already been preprocessed. If that's not the case, please run the data cleaner once.
#     - The call also passes optimized hyperparameters for the neural network. If you want to run the optimizer first, do not pass argument to algorithm_bestparams and enable optimize_optuna. 
#     - There is the option of enabling either oversample or undersample.

Churninator_ = Churninator(path_to_file=root_directory+'/data/clean_data.csv',
                           features2use=features2use,
                           preprocess_file=False,
                           algorithm_bestparams=root_directory+'/data/bestparams_NN.json',
                           optimize_optuna=False,
                           verbose=False,
                           oversample=False,
                           undersample=False,
                           algorithm='NN')

# ## Make predictions and plots
#

y_pred = Churninator_.predict_NN(X_test = Churninator_.X_test) 

Churninator_.plot_ROC(X_test = Churninator_.X_test, y_test = Churninator_.y_test)

cf = Churninator_.compute_confusion_matrix(X_test = Churninator_.X_test, y_test = Churninator_.y_test)




