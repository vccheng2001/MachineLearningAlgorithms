import torch
import torch.nn as nn
import numpy as np

# Takes y coordinates of airfoils as input and predict whether airfoils are real or fake.
# input: 16x200
class Discriminator(nn.Module):
    def __init__(self, input_dim): # 200
        super(Discriminator, self).__init__()
        # build your model here
        # your output should be of dim (batch_size, 1)
        # since discriminator is a binary classifier

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(), # returns 0 or 1
        )
    
    def forward(self, x): # input x: 16x200
                          # output: 16x1
        # define your feedforward pass
        # x = x.view(x.shape[0], -1)
        validity = self.model(x)
        # print(f"Discriminator output shape: {validity.shape}") # 16 x1
        return validity


# output: 16x200
# Takes the normal distributed noise as input and synthesizes the y coor- dinates of airfoils.
class Generator(nn.Module):
    def __init__(self, latent_dim, airfoil_dim):
        super(Generator, self).__init__()
        # build your model here
        # your output should be of dim (batch_size, airfoil_dim)
        # you can use tanh() as the activation for the last layer
        # since y coord of airfoils range from -1 to 1
        # def block(in_feat, out_feat, normalize=True):
        #     layers = [nn.Linear(in_feat, out_feat)]
        #     if normalize:
        #         layers.append(nn.BatchNorm1d(out_feat, 0.8))
        #     layers.append(nn.LeakyReLU(0.2, inplace=True))
        #     return layers

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 200),
            nn.Tanh()
        )
    
    def forward(self, x): # input x: batch_size x latent_dim
        # print(f"Input to Generator Shape: {x.shape}") # 16x16
        # define your feedforward pass
        generated = self.model(x)
        # x = x.view(img.size(0), *img_shape)
        return generated


