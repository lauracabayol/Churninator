"""
plot_feature_importances_seaborn Module
=======================================

This module contains a function to plot feature importances using seaborn.
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

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
