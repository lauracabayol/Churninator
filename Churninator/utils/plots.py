"""
plot_feature_importances_seaborn Module
=======================================

This module contains a function to plot feature importances using seaborn.
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams["font.family"] = "serif"

def plot_feature_importances_seaborn(feature_importance_dict):
    """
    Plots feature importances using seaborn.

    Args:
        feature_importance_dict (dict): Dictionary where keys are feature names and values are their importances.
    
    Returns:
        None
    """
    # Convert the dictionary to a DataFrame
    feature_importance_df = pd.DataFrame(list(feature_importance_dict.items()), columns=['Feature', 'Importance'])
    
    # Sort the DataFrame by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Feature Importances')
    plt.show()


def plot_correlation_matrix(df, columns):
    """
    Plots the correlation matrix for the specified columns in the DataFrame.

    Parameters:
    columns (list): List of column names to include in the correlation matrix.
    """

    # Select the specified columns
    df_selected = df[columns]
    
    # Calculate the correlation matrix
    corr_matrix = df_selected.corr()
    
    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5, linecolor='black')
    plt.title('Correlation Matrix')
    plt.show()

def plot_correlation2exit(df):
    """
    Plots the correlation of features with the target variable 'Exited'.
    """

    features2use = ['CreditScore', 'Age', 'Tenure', 'Balance (EUR)', 'NumberOfProducts', 'IsActiveMember','EstimatedSalary', 'encoded_Country','encoded_sentiments','score','Exited']
    dfplot = df[features2use]

    # Calculate correlations with the target variable
    correlations = dfplot.corr()['Exited'].drop('Exited')
    
    # Plot the correlations
    plt.figure(figsize=(8, 4))
    sns.barplot(x=correlations.index, y=correlations.values, palette='coolwarm')
    plt.title('Correlation with Exit')
    plt.xlabel('Feature')
    plt.ylabel('Correlation')
    plt.xticks(rotation=45)
    plt.show()

