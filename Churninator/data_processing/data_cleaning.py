"""
DataCleaner Module
====================

This module contains the DataCleaner class for cleaning customer data and performing churn prediction analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

class DataCleaner():
    """
    A class for cleaning customer data and performing churn prediction analysis.
    """

    def __init__(self,
                 path_to_data, 
                 labels_to_encode=['Gender', 'Country'], 
                 save_file_path=None, 
                 verbose=False):
        """
        Initializes the data cleaner with the specified parameters.

        Parameters:
        path_to_data (str): Path to the Excel file containing the data.
        labels_to_encode (list): List of labels to encode. Default is ['Gender', 'Country'].
        save_file_path (str): Path to save the cleaned data as a CSV file. Default is None.
        verbose (bool): If True, prints detailed logs. Default is False.
        """
        self.verbose = verbose

        # Load pre-trained model and tokenizer
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model_bert = BertForSequenceClassification.from_pretrained(model_name).to('cpu')
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=model_bert, tokenizer=tokenizer)

        # Load data
        if self.verbose:
            print("Loading data from:", path_to_data)
        self.df = pd.read_excel(path_to_data)
        self.df.fillna({'CustomerFeedback': 'Nothing'}, inplace=True)

        # Encode labels
        if self.verbose:
            print(f"Encoding {labels_to_encode} into discrete numerical values.")
            print("We assume that the surname is not important for churn prediction and dismiss the variable")
            
        if labels_to_encode is not None:
            for lab in labels_to_encode:
                if self.verbose:
                    print(f"Encoding label: {lab}")
                self._label_encoding(lab)

        # Analyze sentiments
        if self.verbose:
            print("Analyzing customer feedback. Classifying feedback into positive, neutral, and negative and encode it into a discrete value")
            
        self.df[['sentiments', 'score']] = self.df['CustomerFeedback'].apply(
            lambda feedback: pd.Series(self._analyze_sentiment_bert(feedback)))
        
        self._label_encoding('sentiments')
        self.df.loc[self.df['CustomerFeedback'] == 'Nothing', ['sentiments', 'encoded_sentiments']] = ['N/A', -99]

        # Select non-string columns
        df_non_string = self.df.select_dtypes(exclude=['object'])

        # Save the cleaned data if a path is provided
        if save_file_path is not None:
            if self.verbose:
                print("Saving cleaned data to:", save_file_path)
            df_non_string.to_csv(save_file_path, header=True, sep=',')
        
        self.df_non_string = df_non_string


    def _label_encoding(self, label, return_mapping=False):
        """
        Encodes a specified label using LabelEncoder.

        Parameters:
        label (str): The label to encode.
        return_mapping (bool): If True, returns the mapping of the encoded labels. Default is False.

        Returns:
        dict: Mapping of the encoded labels if return_mapping is True.
        """
        label_encoder = LabelEncoder()
        self.df[f'encoded_{label}'] = label_encoder.fit_transform(self.df[label].values)
        if return_mapping:
            le_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            return le_mapping

    def _analyze_sentiment_bert(self, text):
        """
        Analyzes the sentiment of a given text using a pre-trained BERT model.

        Parameters:
        text (str): The text to analyze.

        Returns:
        tuple: The sentiment label and score.
        """
        result = self.sentiment_pipeline(text)
        return result[0]['label'], result[0]['score']


