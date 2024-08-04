import torch
import torch.nn as nn

class ChurnNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhidden):
        super(ChurnNet, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for i in range(nhidden):
            self.layers.append(nn.Linear(hidden_dim + i*10, hidden_dim + (i+1)*10))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim + nhidden*10, output_dim)
        
        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()
        
        # Initialize the bias term for the output layer and fix it
        initial_bias = torch.tensor([0.2 / 0.8]).log()  # Set the initial bias
        self.output_layer.bias = nn.Parameter(initial_bias, requires_grad=True)  # Fix the bias
        
    def forward(self, x):
        for layer in self.layers:#, self.dropouts):
            x = layer(x)
            x = torch.relu(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x
