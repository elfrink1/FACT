from eldr.models.autoencoder import *
from eldr.data import *
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.model_selection import train_test_split


def train_ae(input_,\
			 encoder_shape = [100,100,100],\
			 output_dim = 2,\
			 decoder_shape = [100,100,100],\
			 learning_rate = 0.001,\
			 batch_size = 4,\
			 min_epochs = 100,\
			 stopping_epochs = 50,\
			 tol = 0.001,\
			 eval_freq = 1):
	
	
	""" split the dataset """
	x_train, x_val = train_test_split(input_, test_size=0.25)
	
	print(input_.shape, x_train.shape, x_val.shape)
	
	trainloader = DataLoader(Data(x_train), batch_size = batch_size, shuffle = True)
	validloader = DataLoader(Data(x_val), batch_size, shuffle=True)
	
	model = AutoEncoder(x_train.shape[1], output_dim, encoder_shape, decoder_shape)
	
	
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	
	epoch = 0
	best_val_loss = np.inf
	while True:
		epoch_loss = []
		model.train()
		for i, batch in enumerate(trainloader,0):
			input = batch.float()
			model.zero_grad()

			recon = model(input)

			loss = criterion(recon, input)
			epoch_loss.append(loss.item())
			loss.backward()
			optimizer.step()
			
		if epoch%eval_freq == 0:
			valid_loss = []
			with torch.no_grad():
				for i, batch_val in enumerate(validloader,0):
					val = batch_val.float()
					recon = model(val)
					loss = criterion(recon, val)
					valid_loss.append(loss.item())
		val_loss = np.average(valid_loss)
		
		print("Epoch [{}], Train Loss {:.3f}, Valid Loss {:.3f}".format(epoch, np.average(epoch_loss), val_loss))
		if val_loss < best_val_loss - tol:
			best_val_loss = val_loss
			best_epoch = epoch
			print("Saving best model at {}th epoch".format(best_epoch))
			torch.save(model, './Models/' + 'best.pt')
					
		
		
		
		if epoch - best_epoch > stopping_epochs and epoch > min_epochs:
			break
		
		epoch += 1
		
	best = torch.load('./Models/best.pt')

	return best    
				
				
			
		
		
			
		
		
	
	
	
	
	
			 