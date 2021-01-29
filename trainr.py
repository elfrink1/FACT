#script to train the learning low-dimensional representation function

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sys
import os
import json
from types import SimpleNamespace
import argparse
from Model import *



def main(args):
	"""
		Either train the low-dimensional representation learning function and
		visualize the representations
		or load the pretrained model to visualize the representations

	"""

	#load the config file for the corresponding model_type
	config_path = os.path.join('./configs', args.model_type, '.json')
	config = SimpleNamespace(**json.load(open(config_path, 'r')))


	#define the features_path, labels_path, model_dir, and the dataset
	config.features_path = os.path.join(args.data_path, 'X.tsv')
	config.labels_path = os.path.join(args.data_path, 'y.tsv')
	config.model_dir = args.model_dir
	config.dataset = args.dataset


	if args.model_type == 'autoencoder':
		#autoencoder takes input
		input_ = pd.read_csv(config.features_path, sep="\t").to_numpy()
	else:
		input_=None


	if args.train:
		#train the model
		model = Model.Initialize(args.model_type, input_, pretrained_path=None, config=config)
	else:
		#else load the best trained model
		model = Model.Initialize(args.model_type, input_, pretrained_path=args.pretrained_path, config=config)

	#Visualize the low-dimensional representations using the traine model
	#This is to understand the number of clusters
	x = pd.read_csv(config.features_path, sep="\t").to_numpy()
	data_rep = model.Encode(x)
	plt.scatter(data_rep[:,0], data_rep[:,1])
	plt.show()






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
	parser.add_argument('--model_dir',
					  default='./Models',
					  type=str,
					  help='Path to save the trained model')
	parser.add_argument('--data_path',
					  default='./ELDR/Housing/Data',
					  type=str,
					  help='Path of the data to use')
	# parser.add_argument('--xydata',
	# 					action='store_true',
	# 					help='Labels and data stored seperately')
	parser.add_argument('--train',
						action='store_true',
						help='Do you want to train?')
	parser.add_argument('--dataset',
						default='random',
						type=str,
						help='Dataset on which you are training or equivalently exp_name. Trained model will be saved with this name.')

	args = parser.parse_args()
	main(args)