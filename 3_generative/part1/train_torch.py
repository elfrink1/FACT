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

import argparse
import os
import datetime
import statistics
import random

from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from bmnist import bmnist
from mlp_encoder_decoder import MLPEncoder, MLPDecoder
from cnn_encoder_decoder import CNNEncoder, CNNDecoder
from utils import *
import matplotlib.pyplot as plt



class Plotter(object):
    def __init__(self, name):
        self.name = name
        self.bpd = {'train':[], 'valid':[]}

    def update(self, bpds):
        trainL = bpds[0]
        testL = bpds[1]

        self.bpd['train'].append(trainL)
        self.bpd['valid'].append(testL)

    def plot(self):

        iters = range(len(self.bpd['train']))

        plt.plot(iters, self.bpd['train'], c='dodgerblue', label="training bpd")
        plt.plot(iters, self.bpd['valid'], "--", c='red', label="valid bpd")
        plt.xlabel('epoch', fontsize=12)
        plt.ylabel('bpd', fontsize=12)
        plt.title("BPD curve", fontsize=10)
        plt.legend(loc="best", fontsize=12, frameon=False)
        
        # ax2.plot(iters, self.accuracy['train'], c='dodgerblue', label="train accuracy")
        # ax2.plot(iters, self.accuracy['test'], "--", c='red', label="test accuracy")
        # ax2.set_xlabel('steps', fontsize=12)
        # ax2.set_ylabel('Accuracy', fontsize=12)
        # ax2.legend(loc="best", fontsize=12, frameon=False)
        # ax2.set_title("Accuracy Curve", fontsize=10)

        #plt.suptitle("Loss Accuracy curve for default parameters", fontsize=14)
        plt.tight_layout()
        plt.savefig('./' + self.name + '.png')



class VAE(nn.Module):

    def __init__(self, model_name, hidden_dims, num_filters, z_dim, *args,
                 **kwargs):
        """
        PyTorch module that summarizes all components to train a VAE.
        Inputs:
            model_name - String denoting what encoder/decoder class to use.  Either 'MLP' or 'CNN'
            hidden_dims - List of hidden dimensionalities to use in the MLP layers of the encoder (decoder reversed)
            num_filters - Number of channels to use in a CNN encoder/decoder
            z_dim - Dimensionality of latent space
        """
        super().__init__()
        self.z_dim = z_dim

        if model_name == 'MLP':
            self.encoder = MLPEncoder(z_dim=z_dim, hidden_dims=hidden_dims)
            self.decoder = MLPDecoder(z_dim=z_dim, hidden_dims=hidden_dims[::-1])
        else:
            self.encoder = CNNEncoder(z_dim=z_dim, num_filters=num_filters)
            self.decoder = CNNDecoder(z_dim=z_dim, num_filters=num_filters)

        self.recon_loss = nn.BCEWithLogitsLoss(reduction='none')

        self.sigmoid = nn.Sigmoid()

    def forward(self, imgs):
        """
        The forward function calculates the VAE loss for a given batch of images.
        Inputs:
            imgs - Batch of images of shape [B,C,H,W]
        Ouptuts:
            L_rec - The average reconstruction loss of the batch. Shape: single scalar
            L_reg - The average regularization loss (KLD) of the batch. Shape: single scalar
            bpd - The average bits per dimension metric of the batch.
                  This is also the loss we train on. Shape: single scalar
        """

        # Hint: implement the empty functions in utils.py

        self.encoder.zero_grad()
        self.decoder.zero_grad()

        bsz = imgs.shape[0]

        means, log_stds = self.encoder(imgs)

        #print(means, log_stds)

        #print("means {}, log_stds {}".format(means.shape, log_stds.shape))

        sample = sample_reparameterize(means, torch.exp(log_stds))

        #print("bsz {}, sample {}".format(bsz, sample.shape))

        decoded_mean, decoded_log_std = self.decoder(sample)  #B,C,H,W

        #print(decoded_mean.shape, decoded_log_std.shape)

        #print("decoded {} {} {}".format(decoded.shape, decoded.view(bsz, -1).shape, imgs.view(bsz,-1).shape))

        #L_rec = self.recon_loss(decoded.view(bsz,-1), imgs.view(bsz,-1))

        #log_likelihood = log_likelihood_bernoulli(imgs.view(bsz,-1), self.sigmoid(decoded.view(bsz,-1)))
        log_likelihood = log_likelihood_student(imgs.view(bsz,-1), decoded_mean, torch.exp(decoded_log_std))

        L_rec = log_likelihood

        #print("L_rec {}".format(L_rec.shape))
        #L_rec = torch.sum(L_rec, dim=1)
        #print("L_rec {}".format(L_rec.shape))
        L_reg = KLD(means, log_stds)
        #print(L_rec.shape, L_reg.shape)
        elbo = L_rec - L_reg
        #print("elbo {}".format(elbo.shape))
        #bpd = elbo_to_bpd(elbo, decoded.shape)
        #print(bpd.shape)

        tsne_cost = tsne_repel(imgs.view(bsz,-1), sample, self.z_dim, self.device)

        

        #L_rec, L_reg, bpd = torch.mean(L_rec), torch.mean(L_reg), torch.mean(bpd)
        elbo = torch.mean(elbo)
        #print(tsne_cost, elbo)

        cost = imgs.view(bsz,-1).shape[1]*tsne_cost - elbo
        return cost, elbo, tsne_cost #L_rec, L_reg, bpd

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Function for sampling a new batch of random images.
        Inputs:
            batch_size - Number of images to generate
        Outputs:
            x_samples - Sampled, binarized images with 0s and 1s
            x_mean - The sigmoid output of the decoder with continuous values
                     between 0 and 1 from which we obtain "x_samples"
        """
        z = torch.randn(batch_size, self.z_dim).to(self.device)
        x_mean, x_log_std = self.decoder(z)
        #print(x_mean.shape, x_log_std.shape)
        dist = torch.distributions.studentT.StudentT(df=2.0, loc=x_mean, scale=torch.exp(x_log_std))
        x_samples = dist.sample(sample_shape=torch.Size([]))
        x_samples = x_samples.view(-1,1,28,28)
        #print(x_samples.shape)
        

        # z = torch.randn(batch_size, self.z_dim).to(self.device)
        # x_mean = self.sigmoid(self.decoder(z))
        # bsz, c, h, w = x_mean.shape[0],x_mean.shape[1],x_mean.shape[2],x_mean.shape[3]
        # x_mean = x_mean.view(batch_size, -1)
        # x_samples = torch.bernoulli(x_mean)
        # x_samples = x_samples.view(bsz, c, h, w)

        # raise NotImplementedError
        return x_samples, x_mean

    @property
    def device(self):
        """
        Property function to get the device on which the model is.
        """
        return self.decoder.device


def sample_and_save(model, epoch, summary_writer, batch_size=64, log_dir=None):
    """
    Function that generates and saves samples from the VAE.  The generated
    samples and mean images should be saved, and can eventually be added to a
    TensorBoard logger if wanted.
    Inputs:
        model - The VAE model that is currently being trained.
        epoch - The epoch number to use for TensorBoard logging and saving of the files.
        summary_writer - A TensorBoard summary writer to log the image samples.
        batch_size - Number of images to generate/sample
    """
    # Hints:
    # - You can access the logging directory path via summary_writer.log_dir
    # - Use the torchvision function "make_grid" to create a grid of multiple images
    # - Use the torchvision function "save_image" to save an image grid to disk

    samples, means = model.sample(batch_size)
    summary_writer.add_image('gen',make_grid(samples, padding=10), epoch)
    save_image(samples, log_dir +  str(epoch)+'.png', padding=10)


@torch.no_grad()
def test_vae(model, data_loader):
    """
    Function for testing a model on a dataset.
    Inputs:
        model - VAE model to test
        data_loader - Data Loader for the dataset you want to test on.
    Outputs:
        average_bpd - Average BPD
        average_rec_loss - Average reconstruction loss
        average_reg_loss - Average regularization loss
    """

    model.eval()

    # average_bpd = []
    # average_rec_loss = []
    # average_reg_loss = []
    average_elbo = []
    average_cost = []
    average_tsne_cost = []

    for imgs, _ in data_loader:
        imgs = imgs.to(model.device)

        #L_rec, L_reg, bpd  = model(imgs)
        cost, elbo, tsne_cost  = model(imgs)
        average_elbo.append(elbo.item())
        average_cost.append(cost.item())
        average_tsne_cost.append(tsne_cost.item())

        #print(elbo.item(), cost.item(), tsne_cost.item())

        # average_bpd.append(bpd.item())
        # average_rec_loss.append(L_rec.item())
        # average_reg_loss.append(L_reg.item())
        # break
    
    #raise NotImplementedError
    return np.average(average_elbo), np.average(average_cost), np.average(average_tsne_cost)#np.average(average_bpd), np.average(average_rec_loss), np.average(average_reg_loss)


def train_vae(model, train_loader, optimizer):
    """
    Function for training a model on a dataset. Train the model for one epoch.
    Inputs:
        model - VAE model to train
        train_loader - Data Loader for the dataset you want to train on
        optimizer - The optimizer used to update the parameters
    Outputs:
        average_bpd - Average BPD
        average_rec_loss - Average reconstruction loss
        average_reg_loss - Average regularization loss
    """
    model.train()

    # average_bpd = []
    # average_rec_loss = []
    # average_reg_loss = []
    average_cost = []
    average_elbo = []
    average_tsne_cost = []
    for imgs, _ in train_loader:
        imgs = imgs.to(model.device)

        optimizer.zero_grad()

        #L_rec, L_reg, bpd  = model(imgs)
        cost, elbo, tsne_cost  = model(imgs)

        #average_bpd.append(bpd.item())
        #average_rec_loss.append(L_rec.item())
        #average_reg_loss.append(L_reg.item())
        average_cost.append(cost.item())
        average_elbo.append(elbo.item())
        average_tsne_cost.append(tsne_cost.item())

        cost.backward()
        optimizer.step()
        # break
    
    #raise NotImplementedError
    return np.average(average_elbo), np.average(average_cost), np.average(average_tsne_cost)#, np.average(average_reg_loss)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    """
    Main Function for the full training & evaluation loop of a VAE model.
    Make use of a separate train function and a test function for both
    validation and testing (testing only once after training).
    Inputs:
        args - Namespace object from the argument parser
    """
    if args.seed is not None:
        seed_everything(args.seed)

    # Prepare logging
    experiment_dir = os.path.join(
        args.log_dir, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    checkpoint_dir = os.path.join(
        experiment_dir, 'checkpoints')
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    summary_writer = SummaryWriter(experiment_dir)

    # Load dataset
    train_loader, val_loader, test_loader = bmnist(batch_size=args.batch_size,
                                                   num_workers=args.num_workers)

    # Create model
    model = VAE(model_name=args.model,
                hidden_dims=args.hidden_dims,
                num_filters=args.num_filters,
                z_dim=args.z_dim,
                lr=args.lr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #Sample image grid before training starts
    sample_and_save(model, 0, summary_writer, 64, args.log_dir)
    if args.z_dim == 2:
        img_grid = visualize_manifold(model.decoder)

    # Tracking variables for finding best model
    #best_val_bpd = float('inf')
    best_val_elbo = float('inf')
    best_epoch_idx = 0
    print(f"Using device {device}")
    epoch_iterator = (trange(1, args.epochs + 1, desc=f"{args.model} VAE")
                      if args.progress_bar else range(1, args.epochs + 1))

    plotter = Plotter('bpd')
    iteration = 0
    for epoch in epoch_iterator:
        # Training epoch
        train_iterator = (tqdm(train_loader, desc="Training", leave=False)
                          if args.progress_bar else train_loader)
        # epoch_train_bpd, train_rec_loss, train_reg_loss = train_vae(
        #     model, train_iterator, optimizer)

        epoch_train_elbo, train_cost, train_tc = train_vae(
            model, train_iterator, optimizer)


        # Validation epoch
        val_iterator = (tqdm(val_loader, desc="Testing", leave=False)
                        if args.progress_bar else val_loader)
        #epoch_val_bpd, val_rec_loss, val_reg_loss = test_vae(model, val_iterator)

        epoch_val_elbo, val_cost, val_tc = test_vae(model, val_iterator)

        plotter.update([epoch_train_elbo, epoch_val_elbo])

        # Logging to TensorBoard
        # summary_writer.add_scalars(
        #     "BPD", {"train": epoch_train_bpd, "val": epoch_val_bpd}, epoch)
        # print("Epoch {} bpd_train {} bpd_val {}".format(epoch, epoch_train_bpd, epoch_val_bpd))
        # summary_writer.add_scalars(
        #     "Reconstruction Loss", {"train": train_rec_loss, "val": val_rec_loss}, epoch)
        # print("Epoch {} rec_train {} rec_val {}".format(epoch, train_rec_loss, val_rec_loss))
        # summary_writer.add_scalars(
        #     "Regularization Loss", {"train": train_reg_loss, "val": train_reg_loss}, epoch)
        # print("Epoch {} reg_train {} reg_val {}".format(epoch, train_reg_loss, val_reg_loss))
        # print("###")
        print("Epoch {} elbo_train {} elbo_val {}".format(epoch, epoch_train_elbo, epoch_val_elbo))
        print("Epoch {} cost_train {} cost_val {}".format(epoch, train_cost, val_cost))
        print("Epoch {} tc_train {} tc_val {}".format(epoch, train_tc, val_tc))
        print("###")
        # summary_writer.add_scalars(
        #     "ELBO", {"train": train_rec_loss + train_reg_loss, "val": val_rec_loss + val_reg_loss}, epoch)

        if epoch % 5 == 0:
            sample_and_save(model, epoch, summary_writer, 64, args.log_dir)

        # Saving best model
        if epoch_val_elbo < best_val_elbo:
            best_val_elbo = epoch_val_elbo
            best_epoch_idx = epoch
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "epoch.pt"))

    # Load best model for test
    print(f"Best epoch: {best_epoch_idx}. Load model for testing.")
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "epoch.pt")))

    # Test epoch
    test_loader = (tqdm(test_loader, desc="Testing", leave=False)
                   if args.progress_bar else test_loader)
    test_elbo = test_vae(model, test_loader)
    print(f"Test BPD: {test_elbo}")
    #summary_writer.add_scalars("BPD", {"test": test_bpd}, best_epoch_idx)

    plotter.plot()

    # Manifold generation
    if args.z_dim == 2:
        img_grid = visualize_manifold(model.decoder)
        save_image(img_grid, os.path.join(experiment_dir, 'vae_manifold.png'),
                   normalize=False)

    return test_elbo


if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--model', default='MLP', type=str,
                        help='What model to use in the VAE',
                        choices=['MLP', 'CNN'])
    parser.add_argument('--z_dim', default=20, type=int,
                        help='Dimensionality of latent space')
    parser.add_argument('--hidden_dims', default=[512], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "512 256"')
    parser.add_argument('--num_filters', default=32, type=int,
                        help='Number of channels/filters to use in the CNN encoder/decoder.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0. ' + \
                             'For your assignment report, you can use multiple workers (e.g. 4) and do not have to set it to 0.')
    parser.add_argument('--log_dir', default='VAE_logs/', type=str,
                        help='Directory where the PyTorch logs should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))

    args = parser.parse_args()

    main(args)
