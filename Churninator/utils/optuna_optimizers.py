import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim

import json
from Churninator.ml_predictor.network_architecture import ChurnNet

def optimize_rf_with_optuna(X, y, n_trials=100):
    """
    Optimize RandomForestClassifier hyperparameters using Optuna based on ROC AUC score.
    
    Parameters:
    X (pd.DataFrame or np.ndarray): Feature matrix.
    y (pd.Series or np.ndarray): Target vector.
    n_trials (int): Number of trials for optimization. Default is 100.
    
    Returns:
    dict: Best hyperparameters found by Optuna.
    """
    def objective(trial):
        # Define the hyperparameters to tune
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_int('max_features', 1, 100),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }
        
        # Create the model with the current trial's parameters
        rf = RandomForestClassifier(**params, random_state=42)
        
        # Perform cross-validation and calculate the mean ROC AUC score
        score = cross_val_score(rf, X, y, cv=5, scoring='roc_auc').mean()
        
        return score

    # Create a study and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # Print and return the best parameters
    best_params = study.best_params
    print("Best parameters found: ", best_params)
    return best_params

def optimize_gb_with_optuna(X, y, n_trials=100):
    """
    Optimize GradientBoostingClassifier hyperparameters using Optuna based on class 1 accuracy.
    
    Parameters:
    X (pd.DataFrame or np.ndarray): Feature matrix.
    y (pd.Series or np.ndarray): Target vector.
    n_trials (int): Number of trials for optimization. Default is 100.
    
    Returns:
    dict: Best hyperparameters found by Optuna.
    """

    def class_1_accuracy(y_true, y_pred):
        """
        Calculate accuracy for class 1.
        
        Parameters:
        y_true (pd.Series or np.ndarray): True labels.
        y_pred (pd.Series or np.ndarray): Predicted labels.
        
        Returns:
        float: Accuracy for class 1.
        """
        class_1_indices = (y_true == 1)
        return accuracy_score(y_true[class_1_indices], y_pred[class_1_indices])

    def objective(trial):
        # Define the hyperparameters to tune
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'max_features': trial.suggest_int('max_features', 1, X.shape[1]),
        }
        
        # Create the model with the current trial's parameters
        gb = GradientBoostingClassifier(**params, random_state=42)
        
        # Custom scorer for class 1 accuracy
        scorer = make_scorer(class_1_accuracy)
        # Perform cross-validation and calculate the mean class 1 accuracy score
        score = cross_val_score(gb, X, y, cv=5, scoring=scorer).mean()

        
        #score = cross_val_score(gb, X, y, cv=5, scoring='roc_auc').mean()
        
        return score
    # Create a study and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # Print and return the best parameters
    best_params = study.best_params
    print("Best parameters found: ", best_params)
    return best_params


def optimize_nn_with_optuna(train_function, n_trials=100):
    """
    Optimize neural network hyperparameters using Optuna based on ROC AUC score.
    
    Parameters:
    X (pd.DataFrame or np.ndarray): Feature matrix.
    y (pd.Series or np.ndarray): Target vector.
    n_trials (int): Number of trials for optimization. Default is 100.
    
    Returns:
    dict: Best hyperparameters found by Optuna.
    """
    def objective(trial):
        # Define the hyperparameters to tune
        input_dim = 10
        hidden_dim = trial.suggest_int('hidden_dim', low=16, high = 128, step = 8)
        nhidden = trial.suggest_int('nhidden', low=1, high = 5, step = 1)
        learning_rate = trial.suggest_float('learning_rate', low=1e-4, high = 1e-2, step = 1e-4)
        batch_size = trial.suggest_int('batch_size', low=16, high = 128, step = 8)
        num_epochs = trial.suggest_int('num_epochs', low=20, high = 100,step = 10)
        gamma = trial.suggest_float('gamma', low=0, high = 0.1, step = 0.01)

        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'wAdam', 'Adamax'])
        
        # Create the model, criterion, and optimizer
        model = ChurnNet(input_dim, hidden_dim, 1, nhidden)
        criterion = nn.BCELoss()
        
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'wAdam':
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'Adamax':
            optimizer = optim.Adamax(model.parameters(), lr=learning_rate)

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
            
        # Train the model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model, dataloaders = train_function(model=model, 
                               batch_size=batch_size, 
                               learning_rate=learning_rate, 
                               optimizer_name=optimizer_name, 
                               num_epochs=num_epochs)
        
        # Evaluate the model
        model.eval()
        val_loader = dataloaders['val']
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        roc_auc = roc_auc_score(all_labels, all_preds)
        return roc_auc

    # Create a study and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # Print and return the best parameters
    best_params = study.best_params
    print("Best parameters found: ", best_params)
    return best_params

