import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim : int, output_dim : int, hidden_dim : list,
                 num_layers:int, dropout_rate:float=0.):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # Create input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim[0])
        self.hidden_layers = [nn.Linear(hidden_dim[n+1], hidden_dim[n+2]) for n in range(num_layers-2)]
        self.output_layer = nn.Linear(hidden_dim[-1], output_dim)
        self.relu =  nn.ReLU()
        self.droput = nn.DropOut(dropout_rate)
    
    def forward(self, x):

        outputs = self.relu(self.input_layer(x))
        for h_layer in self.hidden_layers:
            outputs = self.relu(self.dropout(h_layer(outputs)))

        outputs = self.output_layer(outputs)

        return outputs

        
