################################################################################
# MIT License
#
# Copyright (c) 2020 Phillip Lippe
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2020
# Date Created: 2020-11-22
################################################################################
from __future__ import division
import torch
import torch.nn as nn
import numpy as np


class MLPEncoder(nn.Module):

	def __init__(self, input_dim=784, hidden_dims=[512], z_dim=20):
		"""
		Encoder with an MLP network and ReLU activations (except the output layer).

		Inputs:
			input_dim - Number of input neurons/pixels. For MNIST, 28*28=784
			hidden_dims - List of dimensionalities of the hidden layers in the network.
						  The NN should have the same number of hidden layers as the length of the list.
			z_dim - Dimensionality of latent vector.
		"""
		super().__init__()

		# For an intial architecture, you can use a sequence of linear layers and ReLU activations.
		# Feel free to experiment with the architecture yourself, but the one specified here is 
		# sufficient for the assignment.
		self.z_dim = z_dim
		self.hidden_dims = hidden_dims
		inp_dim = input_dim
		self.hidden_dims.extend([self.z_dim])
		layers_mean = []
		for i, hidden_dim in enumerate(self.hidden_dims):
			if i != len(self.hidden_dims)-1:
				layers_mean.append(nn.Linear(inp_dim, hidden_dim))
				layers_mean.append(nn.ReLU())
				inp_dim = hidden_dim
			else:
				layers_mean.append(nn.Linear(inp_dim, hidden_dim))

		layers_std = []
		inp_dim = input_dim
		for i, hidden_dim in enumerate(self.hidden_dims):
			if i != len(self.hidden_dims)-1:
				layers_std.append(nn.Linear(inp_dim, hidden_dim))
				layers_std.append(nn.ReLU())
				inp_dim = hidden_dim
			else:
				layers_std.append(nn.Linear(inp_dim, hidden_dim))

		self.layers_mean = nn.ModuleList(layers_mean)
		self.layers_std = nn.ModuleList(layers_std)

	def forward(self, x):
		"""
		Inputs:
			x - Input batch with images of shape [B,C,H,W] and range 0 to 1.
		Outputs:
			mean - Tensor of shape [B,z_dim] representing the predicted mean of the latent distributions.
			log_std - Tensor of shape [B,z_dim] representing the predicted log standard deviation
					  of the latent distributions.
		"""

		# Remark: Make sure to understand why we are predicting the log_std and not std
		x = x.view(x.shape[0], -1)
		mean = x
		log_std = x
		for layer in self.layers_mean:
			mean = layer(mean)

		for layer in self.layers_std:
			log_std = layer(log_std)

		return mean, log_std


class MLPDecoder(nn.Module):

	def __init__(self, z_dim=20, hidden_dims=[512], output_shape=[1, 28, 28]):
		"""
		Decoder with an MLP network.
		Inputs:
			z_dim - Dimensionality of latent vector (input to the network).
			hidden_dims - List of dimensionalities of the hidden layers in the network.
						  The NN should have the same number of hidden layers as the length of the list.
			output_shape - Shape of output image. The number of output neurons of the NN must be
						   the product of the shape elements.
		"""
		super().__init__()
		self.output_shape = output_shape
		out = np.prod(output_shape) #self.output_shape[0]*self.output_shape[1]*self.output_shape[2]
		self.z_dim = z_dim
		self.hidden_dims = hidden_dims
		self.hidden_dims.extend([out])
		inp_dim = self.z_dim
		layers = []

		for i, hidden_dim in enumerate(self.hidden_dims):
			if i != len(self.hidden_dims)-1:
				layers.append(nn.Linear(inp_dim, hidden_dim))
				layers.append(nn.ReLU())
				inp_dim = hidden_dim
			else:
				layers.append(nn.Linear(inp_dim, hidden_dim))


		self.layers_mean = nn.ModuleList(layers)

		layers = []
		inp_dim = self.z_dim
		for i, hidden_dim in enumerate(self.hidden_dims):
			if i != len(self.hidden_dims)-1:
				layers.append(nn.Linear(inp_dim, hidden_dim))
				layers.append(nn.ReLU())
				inp_dim = hidden_dim
			else:
				layers.append(nn.Linear(inp_dim, hidden_dim))

		self.layers_std = nn.ModuleList(layers)


		# For an intial architecture, you can use a sequence of linear layers and ReLU activations.
		# Feel free to experiment with the architecture yourself, but the one specified here is 
		# sufficient for the assignment.
		#raise NotImplementedError

	def forward(self, z):
		"""
		Inputs:
			z - Latent vector of shape [B,z_dim]
		Outputs:
			x - Prediction of the reconstructed image based on z.
				This should be a logit output *without* a sigmoid applied on it.
				Shape: [B,output_shape[0],output_shape[1],output_shape[2]]
		"""

		x = z
		for layer in self.layers_mean:
			x = layer(x)
		mean = x

		x = z
		for layer in self.layers_std:
			x = layer(x)
		log_std = x

		return mean, log_std

	@property
	def device(self):
		"""
		Property function to get the device on which the decoder is.
		Might be helpful in other functions.
		"""
		return next(self.parameters()).device
