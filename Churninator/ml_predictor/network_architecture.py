"""
ChurnNet Module
=======================================

This module contains the class defining the neural network architecture.
"""

import torch
import torch.nn as nn

class ChurnNet(nn.Module):
    """
    A neural network model for predicting customer churn.

    Attributes:
        layers (nn.ModuleList): List of layers in the network.
        output_layer (nn.Linear): The output layer of the network.
        sigmoid (nn.Sigmoid): Sigmoid activation function.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, nhidden):
        """
        Initializes the ChurnNet model.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Dimension of hidden layers.
            output_dim (int): Dimension of the output layer.
            nhidden (int): Number of hidden layers.
        """
        super(ChurnNet, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for i in range(nhidden):
            self.layers.append(nn.Linear(hidden_dim + i * 10, hidden_dim + (i + 1) * 10))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim + nhidden * 10, output_dim)
        
        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()
        
        # Initialize the bias term for the output layer and fix it
        initial_bias = torch.tensor([0.2 / 0.8]).log()  # Set the initial bias
        self.output_layer.bias = nn.Parameter(initial_bias, requires_grad=True)  # Fix the bias

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the forward pass.
        """
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

# Ensure deterministic behavior for reproducibility
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
