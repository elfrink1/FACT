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
	def __init__(self, n_dim, num_clusters, init_deltas=None, use_scaling=False, init_gammas_logit=None):
		super(TGT, self).__init__()
		"""
			TGT class for Transitive Global Translations Algorithm

			n_dim: dimensionality of the input space
			num_clusters: number of clusters in the latent space representation
			init_deltas: deltas to initialize the translationparameters of the TGT
			use_scaling: boolean Flag (set to True to use scaling)
			init_gammas_logit: gammas to initialize the scaling parameters of the TGT

		"""
		self.use_scaling = use_scaling
		if init_deltas is None:
			self.deltas = nn.ParameterList([nn.Parameter(torch.zeros(n_dim)) for _ in range(num_clusters - 1)])
		else:
			self.deltas = nn.ParameterList([nn.Parameter(start_delta) for start_delta in init_deltas])
		
		if self.use_scaling:
			if init_gammas_logit is None:
				self.logit_gammas = nn.ParameterList([nn.Parameter(torch.zeros(n_dim)) for _ in range(num_clusters - 1)])
			else:
				self.logit_gammas = nn.ParameterList([nn.Parameter(start_gl) for start_gl in init_gammas_logit])
	
	def forward(self, x, initial, target, k=None):
		"""x: Init tensor
		initial:  int for initial cluster
		target: int for target cluster
		k: int for sparsity level

		"""
		#for definitions for different initial and target conditions, refer to the original paper
		#and the report
		if self.use_scaling:
			if initial == target:
				d = torch.zeros((1, x.shape[1]))
				logit_g = torch.zeros((1, x.shape[1]))
			elif initial == 0:
				d = self.deltas[target - 1]
				logit_g = self.logit_gammas[target-1]
			elif target == 0:
				d = -1.0 * torch.exp(-self.logit_gammas[initial - 1]) * self.deltas[initial - 1]
				logit_g = -1.0*self.logit_gammas[initial-1]
			else:
				logit_g = self.logit_gammas[target-1] - self.logit_gammas[initial-1]
				d = -1.0 *  torch.exp(-self.logit_gammas[initial - 1]) *self.deltas[initial - 1] + self.deltas[target - 1]
		else:
			if initial == target:
				d = torch.zeros((1, x.shape[1]))
			elif initial == 0:
				d = self.deltas[target - 1]
			elif target == 0:
				d = -1.0 * self.deltas[initial - 1]
			else:
				d = -1.0 * self.deltas[initial - 1] + self.deltas[target - 1]
				
		if k is not None:
			d = truncate(d, k)
			if self.use_scaling:
				logit_g = truncate(logit_g, k)

		if self.use_scaling:
			g = torch.exp(logit_g)
			return g*x + d, d, logit_g
		else:
			return x + d, d
	
	def _init_params(self):
		pass


class Optimizer(object):
	def __init__(self, model, lr=0.001, clip_val=5.0):

		"""
			Optimizer class to train the TGT

			model(nn.Module): TGT class instance
			lr(float): learning rate to train the TGT
			clip_val(float): float value for gradient clipping

		"""
		self.model = model
		self.lr = lr
		self.clip_val = clip_val

	def clip(self, grad):
		return torch.clamp(torch.squeeze(grad), -1.0*self.clip_val, self.clip_val)

	def update(self, index, factor, delta_grad, gamma_grad=None):
		"""
			update the parameters
			index: index of the parameters to update
			factor: (+1, or -1) to update the parameters (again refer to the paper and the report)
			deltas_grad: gradient of the loss w.r.t. delta parameter
			gamma_grad: gradient of the loss w.r.t. gamma parameter
		"""
		self.model.deltas[index - 1].data += factor*self.lr*self.clip(delta_grad)
		if gamma_grad is not None:
			self.model.logit_gammas[index - 1].data += factor*self.lr*self.clip(gamma_grad)
		
	def step(self, initial, target, delta_grad, gamma_grad=None):
		"""
			Gradient descent step

		initial: int (index) of the initial group for TGT
		target: int (index) of the target group for TGT
		deltas_grad: gradient of the loss w.r.t. delta parameter
		gamma_grad: gradient of the loss w.r.t. gamma parameter

		"""
		if initial == 0:
			# self.model.deltas[target-1].data -= self.lr*self.clip(grad)
			self.update(target, -1, delta_grad, gamma_grad)
		elif target == 0:
			# self.model.deltas[initial - 1].data += self.lr*self.clip(grad)
			self.update(initial, 1, delta_grad, gamma_grad)
		else:
			#print("Gradients", grad)
			# self.model.deltas[initial - 1].data += self.lr * 0.5 * self.clip(grad)
			# self.model.deltas[target - 1].data -= self.lr * 0.5 * self.clip(grad)
			self.update(initial, 0.5, delta_grad, gamma_grad)
			self.update(target, -0.5, delta_grad, gamma_grad)


	 
class Explain(object):
	def __init__(self, model, means, centers, use_scaling=False):

		"""
		Explain class for Explaining Low Dimensional Representations

		model (Model class instance)
		means: means of the clusters in the original space
		centers: centers(centroids) of the clusters in the latent space
		use_scaling: boolean Flag (set to True to use scaling in the TGT Algorithm)

		"""
		self.model = model
		self.means = means
		self.centers = centers
		self.use_scaling = use_scaling


	def explain(self, config, k=None):
		"""
			train the TGT Algorithm
			config: Namespace instance describing the settings for training the TGT
			k: int for sparsity level

		"""
		
		#set the relevant settings for training TGT
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


		#initialize the TGT instance
		tgt = TGT(n_input, num_clusters, init_deltas=deltas, use_scaling=self.use_scaling)

		print(list(tgt.parameters()))



		
		#we are not training the r function
		# for param in self.model.model.parameters():
		# 	param.requires_grad = False

		#define the loss_criterion and the optimizer
		criterion = nn.MSELoss(reduction="sum")

		optimizer = Optimizer(tgt, lr=learning_rate, clip_val=clip_val)

		#define variables to track the training process
		iter = 0
		best_iter = 0
		best_loss = np.inf
		best_deltas = None
		best_gammas = None
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

			#perform the TGT step
			if self.use_scaling:
				explained, d, logit_g = tgt(p, initial, target)
			else:
				explained, d = tgt(p, initial, target)

			

			transformed = self.model.Encode_ones(explained.float())
			# print(transformed.shape, t.shape)			
			regularization_term = lambda_global*torch.mean(torch.abs(d))
			if self.use_scaling:
				regularization_term += lambda_global*torch.mean(torch.abs(logit_g))

			#define the compressed sensing based loss function
			loss = criterion(transformed, t) + regularization_term

			#take the gradients of the loss w.r.t parameters
			if self.use_scaling:
				delta_grad, gamma_grad = torch.autograd.grad(loss, [d, logit_g])
			else:
				deltas_grad = torch.autograd.grad(loss, [d])
				delta_grad = deltas_grad[0]
				gamma_grad = None


			if iter == 0:
				ema = loss.item()
			else:
				ema = discount * ema + (1 - discount) * loss.item()

			#save the best deltas and gammas
			if ema < best_loss - tol:
				best_iter = iter
				best_loss = ema
				best_deltas = torch.empty((num_clusters - 1, n_input))
				for i, delta in enumerate(tgt.deltas):
					best_deltas[i,:] = delta.detach()
				if self.use_scaling:
					best_gammas = torch.empty((num_clusters - 1, n_input))
					for i, gamma in enumerate(tgt.logit_gammas):
						best_gammas[i,:] = gamma.detach()
				if verbose:
					print("Retrieving the best deltas...")
					print("iter: {}, ema: {}, initial {}, target {}".format(iter, ema, initial, target))# best_iter, best_loss)); print(best_deltas); print(list(tgt.parameters()))

			#update the TGT parameters
			optimizer.step(initial, target, delta_grad, gamma_grad)
			
			iter += 1
			#break
		#return the best found parameters of the TGT
		if self.use_scaling:
			# Gammas are returned as logits
			return best_deltas, best_gammas, tgt
		else:
			return best_deltas, tgt


	def metrics(self, x, indices, deltas, epsilon, k = None, logit_gammas=None):
		"""
			get the metrics: correctness and coverage
			x: input data 
			indices(list of lists): indices list defining which input belongs to which cluster
			deltas: translation parameters of the TGT
			epsilon(float): hyper-parameter to define the correctness and coverage metrics
			k(int): sparsity level
			logit_gammas: scaling parameters of the TGT

		"""
		if not torch.is_tensor(x):
			x = torch.tensor(x)

		if not torch.is_tensor(deltas):
			deltas = torch.tensor(deltas)

		n_input = x.shape[1]
		num_clusters = len(indices)

		model = self.model

		#initialize the TGT instance
		tgt = TGT(n_input, num_clusters, init_deltas=deltas, init_gammas_logit=logit_gammas, use_scaling=self.use_scaling)

		#get the metrics
		correctness = np.zeros((num_clusters, num_clusters))
		coverage = np.zeros((num_clusters, num_clusters))
		for initial in range(num_clusters):
			for target in range(num_clusters):

					# Get the points in the initial cluster
					x_init = x[indices[initial]]
					
					# Construct the target region in the representation space
					x_target = x[indices[target]]
					
					# Construct the explanation between the initial and target regions
					# if initial == target:
					# 	d = torch.zeros((1, n_input))
					# elif initial == 0:
					# 	d = deltas[target - 1]
					# elif target == 0:
					# 	d = -1.0 * deltas[initial - 1]
					# else:
					# 	d = -1.0 * deltas[initial - 1] + deltas[target - 1]

						
					# if k is not None:
					# 	d = truncate(d, k)
					if self.use_scaling:
						explained, d, logit_g = tgt(x_init, initial, target,k)
					else:
						explained, d = tgt(x_init, initial, target,k)
					
					# Find the representation of the initial points after they have been transformed
					rep_init = model.Encode(explained) #sess.run(rep, feed_dict={X: x_init, D: np.reshape(d, (1, n_input))})
					
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

		#evaluate the epsilon using the self-similarity condition

		model = self.model
		
		input_dim = x.shape[1]
		num_clusters = len(indices)

		a, b = self.metrics(x, indices, torch.from_numpy(np.zeros((num_clusters - 1, input_dim))), epsilon)

		d = np.diagonal(a)
		return np.mean(d), np.min(d), np.max(d)

		#file = open("epsilon.txt","w")
		#file.write(str(np.mean(d)) + " " + str(np.min(d)) + " " + str(np.max(d)))
		#file.close()

	# e_more should be a sparser vector than e_less
	# counts the percentage of e_more's explanation that is in features chosen by e_less
	def similarity(self, e_more, e_less):
		#simialrity metric between the parameters for different sparsity level
		difference = 0
		for i in range(e_more.shape[0]):
			if e_less[i] != 0:
				difference += np.abs(e_more[i])
		return difference / np.sum(np.abs(e_more))


	def apply(self, indices, c1, d_g, num_points = 200):
		#apply the change (perfrorm the TGT from to the points from group c1(int))
		#d_g = deltas
		#num_points(int): how many points to apply the algorithm to
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