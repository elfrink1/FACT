from eldr.models.autoencoder import *
from eldr.models.vae.train_torch import VAE, train_vae, test_vae, Plotter
from eldr.models.vae.utils import *
from eldr.data import Data

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

import os
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
    eval_freq = 1,
       model_dir='/content/FACT/Models/'):
  """ split the dataset """
  x_train, x_val = train_test_split(input_, test_size=0.25)

  print(input_.shape, x_train.shape, x_val.shape)

  trainloader = DataLoader(Data(x_train), batch_size=batch_size, shuffle=True)
  validloader = DataLoader(Data(x_val), batch_size, shuffle=True)

  model = AutoEncoder(x_train.shape[1], output_dim, encoder_shape,
                      decoder_shape)

  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  epoch = 0
  best_val_loss = np.inf
  while True:
    epoch_loss = []
    model.train()
    for i, batch in enumerate(trainloader, 0):
      input = batch.float()
      model.zero_grad()

      recon = model(input)

      loss = criterion(recon, input)
      epoch_loss.append(loss.item())
      loss.backward()
      optimizer.step()

    if epoch % eval_freq == 0:
      valid_loss = []
      with torch.no_grad():
        for i, batch_val in enumerate(validloader, 0):
          val = batch_val.float()
          recon = model(val)
          loss = criterion(recon, val)
          valid_loss.append(loss.item())
    val_loss = np.average(valid_loss)

    print("Epoch [{}], Train Loss {:.3f}, Valid Loss {:.3f}".format(
        epoch, np.average(epoch_loss), val_loss))
    if val_loss < best_val_loss - tol:
      best_val_loss = val_loss
      best_epoch = epoch
      print("Saving best model at {}th epoch".format(best_epoch))
      torch.save(model, model_dir + 'best.pt')

    if epoch - best_epoch > stopping_epochs and epoch > min_epochs:
      break

    epoch += 1

  best = torch.load(model_dir + '/best.pt')

  return best


def train_scvis(
    dataset = "housing",
    features_path="./Reproduction/Housing/Data/X.tsv",
    labels_path="./Reproduction/Housing/Data/y.tsv",
    model_dir='./Models/',
    batch_size=128,
    min_epochs=200,
    stopping_epochs=25,
    tol=0.001,
    eval_freq=1,
    lr=0.001,
):

  # Load dataset.
  data = Data.from_tsv(features_path, labels_path, split=True)

  train_dataset = Data(data.data, labels=data.labels)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1)
  val_loader = DataLoader(Data(data.valD, labels=data.valY),
                          batch_size=batch_size,
                          num_workers=1)

  # Define model and transport to device.
  model = VAE(input_dim=train_dataset[0][0].shape[1],
              output_shape=[1, train_dataset[0][0].shape[1]],
              model_name='MLP',
              encoder_dims=[128, 64, 32],
              decoder_dims=[32, 32, 32, 64, 128],
              z_dim=2,
              lr=lr)

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = model.to(device)

  # Create optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)

  # Tracking variables for finding best model
  best_val_elbo = -1.0*float('inf')
  best_epoch_idx = 0
  iteration = 0

  plotter = Plotter(dataset)

  for epoch in range(1, min_epochs + 1):

    epoch_train_elbo, train_cost, train_tc = train_vae(model, train_loader,
                                                       optimizer)

    epoch_val_elbo, val_cost, val_tc = test_vae(model, val_loader)

    plotter.update([train_cost, val_cost])

    print("Epoch {} elbo_train {} elbo_val {}".format(epoch, epoch_train_elbo,
                                                      epoch_val_elbo))
    print("Epoch {} cost_train {} cost_val {}".format(epoch, train_cost,
                                                      val_cost))
    print("Epoch {} tc_train {} tc_val {}".format(epoch, train_tc, val_tc))
    print("###")

    # Saving best model
    if epoch_val_elbo > best_val_elbo:
      best_val_elbo = epoch_val_elbo
      best_epoch_idx = epoch
      print("Saving the best model at epoch {}".format(best_epoch_idx))
      torch.save(model, os.path.join(model_dir, "scvis_{0}.pt".format(dataset)))

  best = torch.load(os.path.join(model_dir, "scvis_{0}.pt".format(dataset)), map_location=torch.device('cpu'))
  plotter.plot()

  return best
