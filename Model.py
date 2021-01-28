from eldr.models.autoencoder import *
from eldr.models.vae.train_torch import *
from eldr.train import train_ae, train_scvis
from eldr.data import *
from torch.utils.data import Dataset, DataLoader
from types import SimpleNamespace
import json
import os
import sys
from eldr.models.vae.utils import sample_reparameterize

class Model(object):
	def __init__(self, model, model_type):
		"""
		Model class initialzes the model used for learning low-dimensional representations
		model:  trained model (vae or autoencoder)
		model_type: vae or encoder

		Methods of the class:
		a) Encode(input_): maps the input_ to low-dimensional latent space. This is used for the whole dataset.
		b) Encode_ones(input) : maps a batch (input) to low-dimensional latent space
		"""
		self.model = model
		self.model_type = model_type


	@classmethod
	def Initialize(cls, model_type, input_, pretrained_path=None, config=None):
		"""
		Initialize the low-dimensional representation learning model

		model_type: either autoencoder or vae (variational autoencoder)
		input_ : data to train the model on (used in case of autoencoder)
		pretrained_path : path to the pretrained model
		config: Python Namespace object with setting information to train the models

		"""
		if model_type != 'autoencoder' and model_type != 'vae':
			sys.exit("model_type wrong, provide right model type from: [autoencoder, vae]")

		if model_type == 'autoencoder':
			if pretrained_path == None:
				"""
					Train the model and load the best model
				"""
				print("Wait, the model is in training...")
				#Load the config file for the autoencoder
				path = os.path.join('./configs', str(model_type) + '.json')
				config = json.load(open(path, 'r'))
				config = SimpleNamespace(**config)

				print(config)

				"""
				train the model and return the best model.
				"""
				model = train_ae(input_,\
						encoder_shape=config.encoder_shape,\
						output_dim=config.output_dim,\
						decoder_shape=config.decoder_shape,\
						learning_rate=config.learning_rate,\
						batch_size=config.batch_size,\
						min_epochs=config.min_epochs,\
						stopping_epochs=config.stopping_epochs,\
						tol=config.tol,\
						eval_freq=config.eval_freq)

			else:	
				#Use the pretrained model placed at the pretrained_path
				#for now the whole model is saved after training, so 
				#this doesn't require any args while loading.
				print("Loading the pretrained model...")
				model = torch.load(pretrained_path)

		if model_type == 'vae':
			if pretrained_path == None:
				print("Wait, the model is in training")
				model = train_scvis(
									dataset = config.dataset,
    								features_path=config.features_path,
    								labels_path=config.labels_path,
    								model_dir=config.model_dir,
    								batch_size=config.batch_size,
    								min_epochs=config.min_epochs,
    								stopping_epochs=config.stopping_epochs,
    								tol=config.tol,
    								eval_freq=config.eval_freq,
    								lr=config.lr,)

			else:
				print("Loading the pretrained model...")
				model = torch.load(pretrained_path, map_location=torch.device('cpu'))

		return cls(model, model_type)



	def Encode(self, input_):
		"""
			Encode the input_ into low dimensional representation
		"""
		recons = torch.empty(input_.shape[0], 2)
		self.model.eval()
		dl = DataLoader(Data(input_), batch_size = 1, shuffle = False)
		for i, batch in enumerate(dl,0):
				input = batch.float()
				if self.model_type != 'vae':
					recon = self.model.encoder(input)
					recons[i,:] = recon.data.view(-1,2)
				else:
					means, log_stds = self.model.encoder(input)
					recons[i,:] = means.data.view(-1,2)

			
		return recons

	def Encode_ones(self, input):
		if self.model_type != 'vae':
			return self.model.encoder(input)
		else:
			means, log_stds = self.model.encoder(input)
			return means
