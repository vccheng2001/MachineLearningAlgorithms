'''
train and test GAN model on airfoils
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from dataset import AirfoilDataset
from gan import Discriminator, Generator
from utils import *
import numpy as np
from torch.autograd import Variable

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
    lr_dis = 0.0005 # discriminator learning rate
    lr_gen = 0.0005 # generator learning rate
    num_epochs = 60

    # build the model
    dis = Discriminator(input_dim=airfoil_dim).to(device)
    gen = Generator(latent_dim=latent_dim, airfoil_dim=airfoil_dim).to(device)
    print("Discriminator model:\n", dis)
    print("Generator model:\n", gen)

    # define your GAN loss function here
    # you may need to define your own GAN loss function/class
    loss = torch.nn.BCELoss()

    # define optimizer for discriminator and generator separately
    optim_dis = Adam(dis.parameters(), lr=lr_dis)
    optim_gen = Adam(gen.parameters(), lr=lr_gen)

    Tensor = torch.cuda.FloatTensor if device == "cuda:0" else torch.FloatTensor

    
    # train the GAN model
    for epoch in range(num_epochs):
        for n_batch, (local_batch, __) in enumerate(airfoil_dataloader):
            y_real = local_batch.to(device) # 16x200

             # Adversarial ground truths
            valid = Variable(Tensor(y_real.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(y_real.shape[0], 1).fill_(0.0), requires_grad=False)
        
            
            # -----------------
            #  Train Generator
            # -----------------
            optim_gen.zero_grad() 
            # Sample noise as generator input
            # np.random.normal(mean,sd,(output_shape))
            z = Variable(Tensor(np.random.normal(0, 1, (y_real.shape[0], latent_dim))))

            # Generate a batch of images
            gen_imgs = gen(z)

            # Loss measures generator's ability to fool the discriminator
            loss_gen = loss(dis(gen_imgs), valid)

            loss_gen.backward()
            optim_gen.step()

            # # ---------------------
            # #  Train Discriminator
            # # ---------------------
            optim_dis.zero_grad()

            # # Measure discriminator's ability to classify real from generated samples
            real_loss = loss(dis(y_real), valid) # real world
            fake_loss = loss(dis(gen_imgs.detach()), fake) # output from generator 
            loss_dis = (real_loss + fake_loss) / 2

            loss_dis.backward()
            optim_dis.step()

            # print loss while training
            if (n_batch + 1) % 30 == 0:
                print("Epoch: [{}/{}], Batch: {}, Discriminator loss: {}, Generator loss: {}".format(
                    epoch, num_epochs, n_batch, loss_dis.item(), loss_gen.item()))

    # test trained GAN model
    num_samples = 100
    # create random noise 
    noise = torch.randn((num_samples, latent_dim)).to(device)
    # generate airfoils
    gen_airfoils = gen(noise)
    if 'cuda' in device:
        gen_airfoils = gen_airfoils.detach().cpu().numpy()
    else:
        gen_airfoils = gen_airfoils.detach().numpy()

    # plot generated airfoils
    plot_airfoils(airfoil_x, gen_airfoils)


if __name__ == "__main__":
    main()

