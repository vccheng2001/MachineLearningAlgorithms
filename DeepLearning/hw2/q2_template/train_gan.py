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

# function to train the discriminator network
def train_dis(optim_dis, data_real, data_fake, loss, dis, device):
    # create real, fake labels
    batch_size = data_real.shape[0]
    real_label = label_real(batch_size).to(device)
    fake_label = label_fake(batch_size)
    optim_dis.zero_grad()

    # loss for real (from world)
    output_real = dis(data_real)
    loss_real = loss(output_real, real_label)
    # loss for fake (generated) 
    output_fake = dis(data_fake)
    loss_fake = loss(output_fake, fake_label)

    loss_real.backward()
    loss_fake.backward()
    optim_dis.step()
    
    return (loss_real + loss_fake)/2

# function to train the generator network
def train_gen(optim_gen, data_fake, loss, dis):
    batch_size = data_fake.size(0)
    real_label = label_real(batch_size)

    optim_gen.zero_grad()

    output = dis(data_fake)
    loss = loss(output, real_label)

    loss.backward()
    optim_gen.step()

    return loss

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
    lr_dis = 0.001 # discriminator learning rate
    lr_gen = 0.001 # generator learning rate
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
    # optim_dis = torch.optim.SGD(dis.parameters(), lr=lr_dis, momentum=0.9)
    # optim_gen = torch.optim.SGD(dis.parameters(), lr=lr_gen, momentum=0.9)

    Tensor = torch.cuda.FloatTensor if device == "cuda:0" else torch.FloatTensor

    losses_gen = []
    losses_dis = []
    # train the GAN model
    gen.train()
    dis.train()
    for epoch in range(num_epochs):
        loss_gen = 0
        loss_dis = 0
        

        for n_batch, (local_batch, __) in enumerate(airfoil_dataloader):
            y_real = local_batch.to(device) # 16x200

            batch_size = y_real.shape[0]

            for step in range(3):
                # real, fake images
                data_fake = gen(create_noise(batch_size, latent_dim)).detach()
                data_real = y_real
                # train discriminator network
                loss_dis = train_dis(optim_dis, data_real, data_fake, loss,\
                    dis, device)
            # run generator given input noise
            data_fake = gen(create_noise(batch_size, latent_dim))
            # train generator 
            loss_gen = train_gen(optim_gen, data_fake, loss, dis)
          
            #  # Adversarial ground truths
            # valid = Variable(Tensor(y_real.shape[0], 1).fill_(1.0), requires_grad=False)
            # fake = Variable(Tensor(y_real.shape[0], 1).fill_(0.0), requires_grad=False)
            # # -----------------
            # #  Train Generator
            # # -----------------
            # optim_gen.zero_grad() 
            # # Sample noise as generator input
            # # np.random.normal(mean,sd,(output_shape))
            # z = Variable(Tensor(np.random.normal(0, 1, (y_real.shape[0], latent_dim))))
            # # Generate a batch of images
            # gen_imgs = gen(z)
            # # Loss measures generator's ability to fool the discriminator
            # loss_gen = loss(dis(gen_imgs), valid)
            # loss_gen.backward()
            # optim_gen.step()

            # # # ---------------------
            # # #  Train Discriminator
            # # # ---------------------
            # optim_dis.zero_grad()

            # # # Measure discriminator's ability to classify real from generated samples
            # real_loss = loss(dis(y_real), valid) # real world
            # fake_loss = loss(dis(gen_imgs.detach()), fake) # output from generator 
            # loss_dis = (real_loss + fake_loss) / 2

            # loss_dis.backward()
            # optim_dis.step()

            # print loss while training
            if (n_batch + 1) % 30 == 0:
                print("Epoch: [{}/{}], Batch: {}, Discriminator loss: {}, Generator loss: {}".format(
                    epoch, num_epochs, n_batch, loss_dis.item(), loss_gen.item()))
        
        losses_gen.append(loss_gen)
        losses_dis.append(loss_dis)
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

     # plot training loss over epoch
    plt.plot(range(num_epochs),losses_dis, label="discriminator training loss")
    plt.plot(range(num_epochs),losses_gen, label="generator training loss")
    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    plt.legend()
    plt.savefig('training loss')
    plt.show()

if __name__ == "__main__":
    main()

