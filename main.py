# train the explanations for any dataset

import numpy as np 
import pandas as pd
import os
import json
from types import SimpleNamespace
import argparse
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.cluster import KMeans

from Model import *
from eldr.plotter.myplot import *
from eldr.explain.explain_cs import *

#load the model

def load_model(model_type, input=None, pretrained_path=None):
	model = Model.Initialize(model_type = model_type, input_ = input, pretrained_path = pretrained_path)
	return model

def get_xy_data(data_path):
	features_path = os.path.join(data_path, 'X.tsv')
	labels_path = os.path.join(data_path, 'y.tsv')
	x = pd.read_csv(features_path, sep='\t').values
	y = pd.read_csv(labels_path, sep='\t').values
	return x, y

def get_data(features_path, labels_path=None, labels=True):
	features_path = os.path.join(features_path)
	x = pd.read_csv(features_path).to_numpy()
	if labels_path != None and labels == True:
		labels_path = os.path.join(labels_path)
		y = pd.read_csv(labels_path).to_numpy()
	elif labels == True:
		y = x[:,-1]
		x = x[:,:-1]
	return x, y

def plot_difference(path, labels, y):
	fig, ax = plt.subplots(figsize=(10,10))
	sns.boxplot(ax=ax, x=labels, y = np.squeeze(y))
	ax.set(xlabel="Group", ylabel="Label")
	ax.get_figure().savefig(path)

def find_epsilon(Explainer, input_=None, indices=None):
	epsilons = np.linspace(0, 2.0, num=100).tolist()
	for epsilon in epsilons:
		mean_, min_, max_ = Explainer.eval_epsilon(input_, indices, epsilon)
		print("epsilon {}, mean {}, min {}, max {}".format(epsilon, mean_, min_, max_))
		if mean_ >= 0.95 and min_ >= 0.95 and max_>= 0.95:
			print("The epsilon value is {}".format(epsilon))
			return epsilon
	return epsilon

def train(args, Explainer, x=None, epsilon=None, indices=None, exp_mean=None):
	# Columns are:  K, TGT-correctness, TGT-coverage, DBM-correctness, DBM-coverage
	print("Training the TGT and comparing DBM...")
	deltas_path = os.path.join(args.exp_path, 'deltas')
	os.makedirs(os.path.join(args.exp_path, 'deltas'), exist_ok=True)
	K = [1, 3, 5, 7, 9, 11, 13]
	config = SimpleNamespace(**json.load(open('./configs/tgt.json', 'r')))
	config.learning_rate = 0.01
	config.consecutive_steps = 5
	out = np.zeros((len(K), 5))
	input_dim = x.shape[1]
	model = Explainer.model
	use_scaling = Explainer.use_scaling
	c = 0
	for k in K:
		out[c, 0] = k
		best_val = 0.0
		a, b = Explainer.metrics(x, indices, exp_mean, epsilon, k = k)
		out[c, 3] = np.mean(a)
		out[c, 4] = np.mean(b)
		for lg in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
			for trial in range(5):
				config.lambda_global = lg
				print("config ", config, "k ", k)
				if use_scaling:
					deltas, logit_gammas, _ = Explainer.explain(config)
					a, b = Explainer.metrics(x, indices, deltas, epsilon, k = k, logit_gammas=logit_gammas) 
				else:
					deltas, _ = Explainer.explain(config)
					a, b = Explainer.metrics(x, indices, deltas, epsilon, k = k) 
				val = np.mean(a)
				if val > best_val:
					best_val = val
					out[c, 1] = best_val
					out[c, 2] = np.mean(b)
					np.save(os.path.join(deltas_path, "deltas" + str(k) + ".npy"), deltas)
					if use_scaling:
						np.save(os.path.join(deltas_path, "logit_gammas" + str(k) + ".npy"), logit_gammas)
		c += 1
	#saving the metrics output file
	np.savetxt(os.path.join(args.exp_path, "out.csv"), out, delimiter = ",")

def main(args):
	print(args)
	args.exp_path = os.path.join('./experiments', args.exp_name)
	os.makedirs(args.exp_path, exist_ok=True)
	
	if args.xydata:
		x, y = get_xy_data(args.data_path)
	else:
		x, y = get_data(args.data_path)

	model = load_model(args.model_type,
						input=x,
						pretrained_path=args.pretrained_path)

	#get the low-dimensional representation
	data_rep = model.Encode(x)

	kmeans = KMeans(n_clusters = args.num_clusters, random_state=0).fit(data_rep)

	# means, centers, indices = plot_groups(x,
	# 									data_rep.numpy(),
	# 									args.num_clusters,
	# 									kmeans.labels_,
	# 									name = os.path.join(args.exp_path, 'clusters.png'))

	labels = kmeans.labels_
	n = x.shape[0]
	num_clusters = args.num_clusters
	cluster = -1.0 * np.ones((n))
	indices = [[]] * num_clusters
	centers = [[]] * num_clusters
	means = [[]] * num_clusters
	for i in range(num_clusters):
		indices[i] = []
		for j in range(n):
			if labels[j] == i:
				cluster[j] = i
				indices[i].append(j)
		means[i] = np.mean(x[indices[i], :], axis = 0)
		centers[i] = np.mean(data_rep.numpy()[indices[i], :], axis = 0)
	centers = np.array(centers)
	means = np.array(means)

	# plot_difference(os.path.join(args.exp_path, 'labels.png'),
	# 				kmeans.labels_,
	# 				y)
	print("Find the best epsilon...")
	Explainer = Explain(model, means, centers, use_scaling=args.use_scaling)
	epsilon = find_epsilon(Explainer=Explainer,
							input_=x, 
							indices=indices)

	#epsilon = 0.045

	means = torch.tensor(means).float()
	exp_mean = torch.zeros((args.num_clusters - 1, x.shape[1]))
	for i in range(args.num_clusters - 1):
		exp_mean[i, :] = means[i + 1] - means[0]
	train(args, 
		Explainer=Explainer,
		x=x,
		epsilon=epsilon, 
		indices=indices,
		exp_mean=exp_mean)





if __name__ == "__main__":
	parser = argparse.ArgumentParser(
	  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('--model_type',
					  default='vae',
					  type=str,
					  help='Type of model for Learning low dimensional representations',
					  choices=['vae', 'autoencoder'])
	parser.add_argument('--pretrained_path',
					  default='./Models/vae.pt',
					  type=str,
					  help='Path to the trained model')
	parser.add_argument('--data_path',
					  default='./ELDR/Housing/Data',
					  type=str,
					  help='Path of the data to use')
	parser.add_argument('--num_clusters',
					  default=6,
					  type=int,
					  help='Number of Clusters')
	parser.add_argument('--xydata',
						action='store_true',
						help='Labels and data stored seperately')
	parser.add_argument('--exp_name',
						default='Housing',
						type=str,
						help='Name of the experiment. Everything will be saved at ./experiments/$exp_name$')
	parser.add_argument('--use_scaling',
						action='store_true',
						help='Use extended explanations with exponential scaling')

	args = parser.parse_args()
	main(args)
