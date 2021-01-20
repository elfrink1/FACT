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
from torchvision.utils import make_grid
import numpy as np
from scipy.stats import norm
import math
from eldr.models.vae.tsne_helper import *


def sample_reparameterize(mean, std):
	"""
	Perform the reparameterization trick to sample from a distribution with the given mean and std
	Inputs:
		mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
		std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
			  of the distribution
	Outputs:
		z - A sample of the distributions, with gradient support for both mean and std. 
			The tensor should have the same shape as the mean and std input tensors.
	"""

	z = None
	cuda = mean.is_cuda

	epsilon = torch.randn(mean.shape)

	if cuda:
		device = mean.get_device()
		epsilon = epsilon.to(device)

	z = mean + std*epsilon
	return z


def KLD(mean, log_std):
	"""
	Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
	See Section 1.3 for the formula.
	Inputs:
		mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
		log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
	Outputs:
		KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
			  The values represent the Kullback-Leibler divergence to unit Gaussians.
	"""
	KLD = 0.5*torch.sum(torch.exp(2*log_std) + mean*mean - 1.0 - 2*log_std, dim=-1)
	return KLD


def log_likelihood_student(x, mean, std, df=2.0):
	dist = torch.distributions.studentT.StudentT(df=df, loc=mean, scale=std)
	return torch.sum(dist.log_prob(x), dim=1)

def log_likelihood_bernoulli(x, probs):
	dist = torch.distributions.multinomial.Multinomial(probs=probs)
	return dist.log_prob(x)


def tsne_repel(x, z, z_dim, device):
	bsz = x.shape[0]
	nu = z_dim - 1
	p = compute_transition_probability(x.cpu().numpy(), perplexity=10)
	p = torch.from_numpy(p).to(device)
	sum_y = torch.sum(torch.square(z), dim=1)
	num = (-2.0*z@z.transpose(0,1) + sum_y + sum_y.unsqueeze(1))/nu
	
	p = p + 0.1/bsz

	p = p/(torch.sum(p, dim=1).unsqueeze(1))

	num = torch.pow(1.0 + num, -(nu + 1.0)/2.0)
	attraction = -1.0*torch.sum(p*torch.log(num))

	den = torch.sum(num, dim=1) - 1
	repellant = torch.sum(torch.log(den))


	return (repellant + attraction)/bsz


def elbo_to_bpd(elbo, img_shape):
	"""
	Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
	Inputs:
		elbo - Tensor of shape [batch_size]
		img_shape - Shape of the input images, representing [batch, channels, height, width]
	Outputs:
		bpd - The negative log likelihood in bits per dimension for the given image.
	"""
	term = img_shape[1]*img_shape[2]*img_shape[3]
	bpd = (elbo*math.log(math.e, 2))/term
	return bpd


@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
	"""
	Visualize a manifold over a 2 dimensional latent space. The images in the manifold
	should represent the decoder's output means (not binarized samples of those).
	Inputs:
		decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
		grid_size - Number of steps/images to have per axis in the manifold.
					Overall you need to generate grid_size**2 images, and the distance
					between different latents in percentiles is 1/(grid_size+1)
	Outputs:
		img_grid - Grid of images representing the manifold.
	"""

	## Hints:
	# - You can use scipy's function "norm.ppf" to obtain z values at percentiles.
	# - Use the range [0.5/(grid_size+1), 1.5/(grid_size+1), ..., (grid_size+0.5)/(grid_size+1)] for the percentiles.
	# - torch.meshgrid might be helpful for creating the grid of values
	# - You can use torchvision's function "make_grid" to combine the grid_size**2 images into a grid
	# - Remember to apply a sigmoid after the decoder

	img_grid = None
	x_range = (np.arange(0, grid_size+1)+0.5)/(grid_size+1)
	y_range = x_range

	z_s = []

	for x in x_range:
		for y in y_range:
			z = [norm.ppf(x), norm.ppf(y)]
			z_s.append(z)

	latents = torch.from_numpy(np.array(z_s)).float().to(decoder.device)
	print(latents.shape)

	images = decoder(latents)
	print(images.shape)

	img_grid = make_grid(images, nrow = grid_size+1, padding=10)
	return img_grid
