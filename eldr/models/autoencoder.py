import numpy as nn
import torch
import torch.nn as nn


class MLP(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dims=[]):
        super(MLP, self).__init__()
        
        hidden_dims.extend([output_dim])
        
        layers = []
        prev_hidden = input_dim
        for i, hidden in enumerate(hidden_dims):
            if i != len(hidden_dims)-1:
                layers.append(nn.Linear(prev_hidden, hidden))
                layers.append(nn.LeakyReLU())
                prev_hidden = hidden
            else:
                layers.append(nn.Linear(prev_hidden, hidden))
                
        self.layers = nn.ModuleList(layers)
        self.layers.apply(self._init_weights)
        
    def forward(self, input):
        out = input
        for layer in self.layers:
            out = layer(out)
        return out
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            module.bias.data.fill_(0.0)
            
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, encoder_hidden_dims=[], decoder_hidden_dims=[]):
        super(AutoEncoder, self).__init__()
        
        self.encoder = MLP(input_dim, output_dim, encoder_hidden_dims)
        self.decoder = MLP(output_dim, input_dim, decoder_hidden_dims)
        
        
    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return decoded
        
        
        