import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from eldr.train import *
from eldr.data import *
from sklearn.metrics.pairwise import euclidean_distances
from eldr.misc import truncate

class TGT(nn.Module):
	def __init__(self, n_dim, num_clusters, init_deltas=None, use_scaling=False):
		super(TGT, self).__init__()
		self.use_scaling = use_scaling
		if init_deltas is None:
			self.deltas = nn.ParameterList([nn.Parameter(torch.zeros(n_dim))])
		else:
			self.deltas = nn.ParameterList([nn.Parameter(start_delta) for start_delta in init_deltas])
		
		if self.use_scaling:
			self.gammas = nn.ParameterList([nn.Parameter(torch.ones(n_dim))])
	
	def forward(self, x, cluster, target):
		"""x  Tensor with shape [Batch size, n_dim]
		cluster: (list of) int(s) for which delta to use 
			(use a list if the batch is not homogeneous)"""
# 		print(x)
		initial = cluster
		if initial == 0:
			d = self.deltas[target - 1]
		elif target == 0:
			d = -1.0 * self.deltas[initial - 1]
		else:
			d = -1.0 * self.deltas[initial - 1] + self.deltas[target - 1]

		if self.use_scaling:
			return self.gammas[cluster]*x + self.deltas[cluster]
		else:
			return x + d, d
	
	def _init_params(self):
		pass


class Optimizer(object):
	def __init__(self, model, lr=0.001, clip_val=5.0):
		self.model = model
		self.lr = lr
		self.clip_val = clip_val

	def clip(self, grad):
		return torch.clip(grad, -1.0*self.clip_val, self.clip_val)

	def step(self, initial, target):
		if initial == 0:
			self.model.deltas[target-1].data -= self.lr*self.clip(self.model.deltas[target-1].grad)
		elif target == 0:
			self.model.deltas[initial - 1].data += self.lr*self.clip(self.model.deltas[initial-1].grad)
		else:
			#print("Gradients", self.model.deltas[initial-1].grad, self.model.deltas[target-1].grad)
			self.model.deltas[initial - 1].data += self.lr * 0.5 * self.clip(self.model.deltas[initial-1].grad)
			self.model.deltas[target - 1].data -= self.lr * 0.5 * self.clip(self.model.deltas[target-1].grad)


	 
class Explain(object):
	def __init__(self, model, means, centers):
		self.model = model
		self.means = means
		self.centers = centers

	def explain(self, config):#model, means, centers):
		

		lambda_global = config.lambda_global
		init_mode = config.init_mode
		consecutive_steps = config.consecutive_steps
		learning_rate = config.learning_rate
		clip_val = config.clip_val
		min_iters = config.min_iters
		stopping_iters = config.stopping_iters
		tol = config.tol
		discount = config.discount
		verbose = config.verbose
		
		num_clusters = self.means.shape[0]
		n_input = self.means.shape[1]
		n_output = self.centers.shape[1]


		x_means = Data(self.means)
		y_means = Data(self.centers)


		print(init_mode)

		# Initialize the deltas
		if init_mode == "zero":
			deltas = torch.zeros((num_clusters - 1, n_input)) #Row i is the explanation for "Cluster 0 to Cluster i + 1"
		elif init_mode == "mean":
			deltas = torch.zeros((num_clusters - 1, n_input))
			for i in range(1, num_clusters):
				deltas[i - 1,:] = x_means[i] - x_means[0]


		tgt = TGT(n_input, num_clusters, init_deltas=deltas)

		print(list(tgt.parameters()))

		#print(x_means, y_means)



		
		#we are not training the r function
		for param in self.model.model.parameters():
			param.requires_grad = False

		criterion = nn.MSELoss()

		optimizer = Optimizer(tgt, lr=learning_rate, clip_val=clip_val)


		iter = 0
		best_iter = 0
		best_loss = np.inf
		best_deltas = None
		ema = None
		while True:
		
			# Stopping condition
			if iter - best_iter > stopping_iters and iter > min_iters:
				break

			# Choose the initial and target cluster
			if iter % consecutive_steps == 0:
				initial, target = np.random.choice(num_clusters, 2, replace = False)

			# point and target
			p = x_means[initial]
			t = y_means[target]

			

			tgt.zero_grad()
			explained, d = tgt(p, initial, target)

			

			transformed = self.model.Encode(explained)

			#print(p, t, d, transformed, d.requires_grad)

			loss = criterion(transformed, t) + lambda_global*torch.mean(torch.abs(d))

			#print(criterion(transformed, t).item(), lambda_global*torch.mean(torch.abs(d)))
			
			loss.backward()

			if iter == 0:
				ema = loss.item()
			else:
				ema = discount * ema + (1 - discount) * loss.item()

			print("iter: {}, ema: {}, initial: {}, target: {}".format(iter, ema, initial, target))

			if ema < best_loss - tol:
				best_iter = iter
				best_loss = ema
				best_deltas = torch.empty((num_clusters - 1, n_input))
				for i, param in enumerate(tgt.parameters()):
					best_deltas[i,:] = param.detach()
				if verbose:
					print("Retrieving the best deltas...")
					print("iter: {}, ema: {}".format(iter, ema))

			optimizer.step(initial, target)
			
			iter += 1

		return best_deltas, tgt


	def metrics(self, x, indices, deltas, epsilon, k = None):
		n_input = x.shape[1]
		num_clusters = len(indices)

		model = self.model

		# Define the objective function

		correctness = np.zeros((num_clusters, num_clusters))
		coverage = np.zeros((num_clusters, num_clusters))
		for initial in range(num_clusters):
			for target in range(num_clusters):

					# Get the points in the initial cluster
					x_init = x[indices[initial]]
					
					# Construct the target region in the representation space
					x_target = x[indices[target]]
					
					# Construct the explanation between the initial and target regions
					if initial == target:
						d = torch.zeros((1, n_input))
					elif initial == 0:
						d = deltas[target - 1]
					elif target == 0:
						d = -1.0 * deltas[initial - 1]
					else:
						d = -1.0 * deltas[initial - 1] + deltas[target - 1]
						
					if k is not None:
						d = truncate(d, k)
					
					# Find the representation of the initial points after they have been transformed
					rep_init = model.Encode(x_init+d) #sess.run(rep, feed_dict={X: x_init, D: np.reshape(d, (1, n_input))})
					
					# Find the representation of the target points without any transformation
					rep_target = model.Encode(x_target) #sess.run(rep, feed_dict={X: x_target, D: np.zeros((1, n_input))})
					
					# Calculate pairwise l2 distance
					dists = euclidean_distances(rep_init, Y = rep_target)
					
					# Find which pairs of points are within epsilon of each other
					close_enough = 1.0 * (dists <= epsilon)
					
					if initial == target:
						# In this setting, every point is similar enough to itself
						threshold = 2.0
					else:
						threshold = 1.0

					correctness[initial, target] = np.mean(1.0 * (np.sum(close_enough, axis = 1) >= threshold))
					coverage[initial, target] = np.mean(1.0 * (np.sum(close_enough, axis = 0) >= threshold))

		return correctness, coverage


	def eval_epsilon(self, x, indices, epsilon):

		model = self.model
		
		input_dim = x.shape[1]
		num_clusters = len(indices)

		a, b = self.metrics(x, indices, np.zeros((num_clusters - 1, input_dim)), epsilon)

		d = np.diagonal(a)

		file = open("epsilon.txt","w")
		file.write(str(np.mean(d)) + " " + str(np.min(d)) + " " + str(np.max(d)))
		file.close()

	# e_more should be a sparser vector than e_less
	# counts the percentage of e_more's explanation that is in features chosen by e_less
	def similarity(self, e_more, e_less):
		difference = 0
		for i in range(e_more.shape[0]):
			if e_less[i] != 0:
				difference += np.abs(e_more[i])
		return difference / np.sum(np.abs(e_more))


	def apply(self, indices, c1, d_g, num_points = 200):


		model = self.model
		x = self.means
		y = self.centers

		# Visualize the data
		fig, ax = plt.subplots(figsize=(20, 10))

		plt.subplot(2, 1, 1)
		plt.scatter(y[:, 0], y[:, 1], s = 12)
		
		# Sample num_points in cluster c1
		indices_c1 = np.random.choice(indices[c1], num_points, replace = False)

		points_c1 = x[indices_c1]
		
		d = np.zeros((1, x.shape[1]))
		
		# Plot the chosen points before perturbing them
		y_c1 = model.Encode(points_c1 + d) #sess.run(rep, feed_dict={X: points_c1, D: d})
		plt.scatter(y_c1[:,0], y_c1[:,1], marker = "v", c = "green", s = 64)

		# Plot the chosen points after perturbing them
		y_c1 = model.Encode(points_c1 + d_g) #sess.run(rep, feed_dict={X: points_c1, D: d_g})
		plt.scatter(y_c1[:,0], y_c1[:,1], marker = "v", c = "red", s = 64)
		
		plt.subplot(2, 1, 2)
		
		feature_index = np.array(range(d_g.shape[1]))
		plt.scatter(feature_index, d_g, label = "Explantion - Change per Dataset Feature", marker = "x")

		plt.show()
	
		plt.close()