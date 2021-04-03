'''
train and test VAE model on airfoils
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from torch.nn import functional as F


from dataset import AirfoilDataset
from vae import VAE
from utils import *

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_func(recon_x, x, mu, logvar, airfoil_dim):
    # recon, x: 16x 200
    # mu, logvar: 16x16
  
    mse = nn.MSELoss(reduction="sum")
    MSE = mse(recon_x, x.view(-1, airfoil_dim))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD

def main():
    # check if cuda available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # define dataset and dataloader
    dataset = AirfoilDataset()
    airfoil_x = dataset.get_x()
    airfoil_dim = airfoil_x.shape[0]
    airfoil_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # hyperparameters
    latent_dim = 16 # please do not change latent dimension
    lr = 0.001      # learning rate
    num_epochs = 30

    # build the model
    vae = VAE(airfoil_dim=airfoil_dim, latent_dim=latent_dim).to(device)
    print("VAE model:\n", vae)

    # define optimizer for discriminator and generator separately
    optim = Adam(vae.parameters(), lr=lr)
    
    training_losses = []

    # train the VAE model
    for epoch in range(num_epochs):
        vae.train()
        print(f"epoch #{epoch}")
        train_loss = 0
        for n_batch, (local_batch, __) in enumerate(airfoil_dataloader):
            y_real = local_batch.to(device)

            optim.zero_grad()
            vae.zero_grad()

            # do VAE
            recon_batch, mu, logvar = vae(y_real)
            # calculate training VAE loss
            loss = loss_func(recon_batch, y_real, mu, logvar, airfoil_dim)
            # accum train loss
            train_loss += loss.item()

            optim.zero_grad()
            loss.backward()
            optim.step()
            
            # print loss while training
            if (n_batch + 1) % 30 == 0:
                print("Epoch: [{}/{}], Batch: {}, loss: {}".format(
                    epoch, num_epochs, n_batch, loss.item()))

        # append training loss for each epoch 
        training_losses.append(train_loss/n_batch)
    # test trained VAE model
    num_samples = 100

    # reconstuct airfoils
    real_airfoils = dataset.get_y()[:num_samples]
    recon_airfoils, __, __ = vae(torch.from_numpy(real_airfoils).to(device))
    if 'cuda' in device:
        recon_airfoils = recon_airfoils.detach().cpu().numpy()
    else:
        recon_airfoils = recon_airfoils.detach().numpy()
    
    # randomly synthesize airfoils
    noise = torch.randn((num_samples, latent_dim)).to(device)   # create random noise 
    gen_airfoils = vae.decode(noise)
    if 'cuda' in device:
        gen_airfoils = gen_airfoils.detach().cpu().numpy()
    else:
        gen_airfoils = gen_airfoils.detach().numpy()

    # plot real/reconstructed/synthesized airfoils

    print(airfoil_x.shape, real_airfoils.shape)
    print(airfoil_x.shape, recon_airfoils.shape)
    print(airfoil_x.shape, gen_airfoils.shape)


    # (200,) (100, 200)
    # (200,) (100, 16) # should be (100, 200)
    # (200,) (100, 200)
    plot_airfoils(airfoil_x, real_airfoils)
    plot_airfoils(airfoil_x, recon_airfoils)
    plot_airfoils(airfoil_x, gen_airfoils)
    

    # plot training loss over epoch
    plt.plot(range(num_epochs),training_losses, label="training loss")
    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    plt.legend()
    plt.savefig('training loss')
    plt.show()

if __name__ == "__main__":
    main()