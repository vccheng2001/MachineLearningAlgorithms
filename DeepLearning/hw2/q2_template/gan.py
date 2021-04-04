import torch
import torch.nn as nn
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        # build your model here
        # your output should be of dim (batch_size, 1)
        # since discriminator is a binary classifier

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        # define your feedforward pass
        print("X SHAPE", x.shape)
        x = x.view(x.shape[0], -1)
        validity = self.model(x)
        print(f"Discriminator output shape: {validity.shape}")
        return validity


class Generator(nn.Module):
    def __init__(self, latent_dim, airfoil_dim):
        super(Generator, self).__init__()
        # build your model here
        # your output should be of dim (batch_size, airfoil_dim)
        # you can use tanh() as the activation for the last layer
        # since y coord of airfoils range from -1 to 1
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 64, normalize=False),
            nn.Linear(64, 200),
            nn.Tanh()
        )
    
    def forward(self, x):
        # define your feedforward pass
        x = self.model(x)
        # x = x.view(img.size(0), *img_shape)
        return x


