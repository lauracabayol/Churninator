# 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

class data_cleaner():
    """
    A class for cleaning customer data and performing churn prediction analysis.
    """

    def __init__(self, path_to_data, 
                 labels_to_encode=['Gender', 'Country'], 
                 save_file_path=None, 
                 verbose=False,
                 make_plots=False):
        """
        Initializes the data cleaner with the specified parameters.

        Parameters:
        path_to_data (str): Path to the Excel file containing the data.
        labels_to_encode (list): List of labels to encode. Default is ['Gender', 'Country'].
        save_file_path (str): Path to save the cleaned data as a CSV file. Default is None.
        verbose (bool): If True, prints detailed logs. Default is False.
        """
        self.verbose = verbose

        # Load data
        if self.verbose:
            print("Loading data from:", path_to_data)
        self.df = pd.read_excel(path_to_data)
        self.df.fillna({'CustomerFeedback': 'Nothing'}, inplace=True)

        # Encode labels
        if self.verbose:
            print(f"Encoding {labels_to_encode} into descreate numerical values.")
            print("We asssume that the surname is not important for churn prediction and dismiss the variable")
            
        if labels_to_encode is not None:
            for lab in labels_to_encode:
                if self.verbose:
                    print(f"Encoding label: {lab}")
                self._label_encoding(lab)

        # Analyze sentiments
        if self.verbose:
            print("Analyzing customer feedback. Classifying feedback into positive, neutral, and negative and encode it into a descreate value")
        sentiments = [self._analyze_sentiment_vader(feedback) for feedback in self.df.CustomerFeedback]
        self.df['sentiments'] = sentiments
        self._label_encoding('sentiments')

        # Select non-string columns
        df_non_string = self.df.select_dtypes(exclude=['object'])

        # Save the cleaned data if a path is provided
        if save_file_path is not None:
            if self.verbose:
                print("Saving cleaned data to:", save_file_path)
            df_non_string.to_csv(save_file_path, header=True, sep=',')
        
        self.df_non_string = df_non_string

        if make_plots:
            self.plot_correlation2exit()
            self.histogram_by_exited()

    def _label_encoding(self, label, return_mapping=False):
        """
        Encodes a specified label using LabelEncoder.

        Parameters:
        label (str): The label to encode.
        """
        label_encoder = LabelEncoder()
        self.df[f'encoded_{label}'] = label_encoder.fit_transform(self.df[f'{label}'].values)
        if return_mapping:
            le_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            return le_mapping


    def _analyze_sentiment_vader(self, text):
        """
        Analyzes the sentiment of a given text using VADER. 
        The thresholds for classifying the sentiment as positive, negative, or neutral are arbitrary 
        but have been found to work well in practice. Here are the commonly used thresholds:

        - Positive: A compound score greater than or equal to 0.05
        - Negative: A compound score less than or equal to -0.05
        - Neutral: A compound score between -0.05 and 0.05 

        Parameters:
        text (str): The text to analyze.

        Returns:
        str: The sentiment classification ('positive', 'negative', or 'neutral').
        """
        sia = SentimentIntensityAnalyzer()
        sentiment_score = sia.polarity_scores(text)
        
        # Classify sentiment
        if sentiment_score['compound'] >= 0.05:
            return 'positive'
        elif sentiment_score['compound'] <= -0.05:
            return 'negative'
        else:
            return 'neutral'       

    def plot_correlation_matrix(self, columns):
        """
        Plots the correlation matrix for the specified columns in the DataFrame.

        Parameters:
        columns (list): List of column names to include in the correlation matrix.
        """
        if self.verbose:
            print("Plotting correlation matrix for columns:", columns)
        
        # Select the specified columns
        df_selected = self.df_non_string[columns]
        
        # Calculate the correlation matrix
        corr_matrix = df_selected.corr()
        
        # Create a heatmap of the correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5, linecolor='black')
        plt.title('Correlation Matrix')
        plt.show()

    def plot_correlation2exit(self):
        """
        Plots the correlation of features with the target variable 'Exited'.
        """
        if self.verbose:
            print("Plotting correlation with exit status")
        
        # Calculate correlations with the target variable
        correlations = self.df_non_string.corr()['Exited'].drop('Exited')
        
        # Plot the correlations
        plt.figure(figsize=(10, 6))
        sns.barplot(x=correlations.index, y=correlations.values, palette='coolwarm')
        plt.title('Correlation with Exit')
        plt.xlabel('Feature')
        plt.ylabel('Correlation')
        plt.xticks(rotation=45)
        plt.show()

    def histogram_by_exited(self):
        """
        Plots a histogram of sentiment distribution by exit status.
        """
        if self.verbose:
            print("Plotting histogram of sentiment distribution by exit status")

        # Get mapping of encoded sentiments
        mapping = self._label_encoding('sentiments', return_mapping=True)
        mapping = dict(sorted(mapping.items(), key=lambda item: item[1]))
        

        # Create the histogram plot
        fig, ax = plt.subplots(1,1) 
        sns.histplot(self.df_non_string, x='encoded_sentiments', hue='Exited', kde=True, palette='coolwarm', multiple='stack')

        # Set plot title and labels
        ax.set_title('Feedback Distribution by Exit Status')
        ax.set_xlabel('Feedback (0: Negative, 1: Neutral, 2: Positive)')
        ax.set_ylabel('Frequency')

        ax.set_xticks(np.arange(0,3,1))
        ax.set_xticklabels(mapping.keys(), rotation='vertical', fontsize=18)

        # Modify legend
        handles, labels = plt.gca().get_legend_handles_labels()
        labels = ['Exited', 'Not Exited']
        ax.legend(handles, labels)

        # Show plot
        plt.show()