import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

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
    Optimize GradientBoostingClassifier hyperparameters using Optuna based on ROC AUC score.
    
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
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'max_features': trial.suggest_int('max_features', 1, 100),
        }
        
        # Create the model with the current trial's parameters
        gb = GradientBoostingClassifier(**params, random_state=42)
        
        # Perform cross-validation and calculate the mean ROC AUC score
        score = cross_val_score(gb, X, y, cv=5, scoring='roc_auc').mean()
        
        return score

    # Create a study and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # Print and return the best parameters
    best_params = study.best_params
    print("Best parameters found: ", best_params)
    return best_params
