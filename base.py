import torch
import torch.nn as nn
import numpy as np

def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    Source:
    Implementation of third assignment fro the DL2020 course at the University of Amsterdam.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """
    std = torch.exp(log_std)
    ones = torch.ones_like(mean)
    pre_sum = 0.5*(torch.square(std) + torch.square(mean) - ones - 2*log_std)
    KLD = torch.sum(pre_sum, dim=-1)
    return KLD

class MLP(nn.Module):
    def __init__(self, dim_list):
        """ Multi layer perceptron with LeakyReLU activation functions.
        -dim_list = [input dimension, *hidden dimensions, output dimension]"""
        super(MLP, self).__init__()
        layer_list = []
        for i in range(1, len(dim_list)):
            in_dim = dim_list[i - 1]
            out_dim = dim_list[i]
            layer_list.append(nn.Linear(in_dim, out_dim))
            if not i == len(dim_list):
                layer_list.append(nn.LeakyReLU())
        self.seq = nn.Sequential(*layer_list)
    
    def forward(self, x):
        return self.seq(x)

class VAE(nn.Module):
    def __init__(self, encoder_shape, decoder_shape):
        super(VAE, self).__init__()
        self.encoder = MLP(encoder_shape)
        self.decoder = MLP(decoder_shape)
    
    def forward(self, x):
        """ Returns reconstruction of the input x aswell as the KL divergence.
        x shape: [Batch size, input dimension]"""

        # Encode
        latent_output = self.encoder(x)
        means, log_stds = torch.chunk(latent_output, 2 , 1)
        stds = torch.exp(log_stds)

        # Reparameterization trick
        noise = torch.rand_like(stds)
        latents = means + stds*noise

        # Decode
        reconstruction_x = self.decoder(latents)

        # Calculate divergence
        divergence = KLD(means, log_stds)

        return reconstruction_x, divergence

class TGT(nn.Module):
    def __init__(self, n_dim, num_clusters, init_deltas=None, use_scaling=False):
        super(TGT, self).__init__()
        self.use_scaling = use_scaling
        if init_deltas is None:
            self.deltas = nn.ParameterList([nn.Parameter(torch.zeros(n_dim))])
        else:
            self.deltas = nn.ParameterList([nn.Parameter(start_delta) for start_delta in init_deltas])
        
        if use_scaling:
            self.gammas = nn.ParameterList([nn.Parameter(torch.ones(n_dim))])
    
    def forward(self, x, cluster):
        """x  Tensor with shape [Batch size, n_dim]
        cluster: (list of) int(s) for which delta to use 
            (use a list if the batch is not homogeneous)"""
        if use_scaling:
            return self.gammas[cluster]*x + self.deltas[cluster]
        else:
            return x + self.deltas[cluster]


        


        
