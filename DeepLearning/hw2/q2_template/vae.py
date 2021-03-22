import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        # build your model here
        # your model should output a predicted mean and a predicted std of the encoding
        # both should be of dim (batch_size, latent_dim)
        '''
        UIUC Airfoil Coordinates Database includes coordinates for nearly 1,600 airfoils.
        Since number of points in each sample differ, we first process all airfoils to
        have 200 points and share the same x coordinates via spline interpo- lation.
        Also, all y coordinates are rescaled to [-1,1]. The processed airfoils are 
        shown in Fig. 7a. Therefore, only y coordinates of each airfoil is used to train and test generative models.
        '''

        self.input_dim = input_dim
        self.latent_dim = latent_dim 

        # define layers: 1600->200
        self.fc = nn.Linear(input_dim, latent_dim)
        
    
    def forward(self, x):
        # define your feedforward pass
        self.encoded = self.fc(x)
     

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        # build your model here
        # your output should be of dim (batch_size, output_dim)
        # you can use tanh() as the activation for the last layer
        # since y coord of airfoils range from -1 to 1

        self.latent_dim = latent_dim
        self.output_dim = output_dim 
        # define layer: 200->1600
        self.fc = nn.Linear(latent_dim, output_dim)
    
    def forward(self, x):
        # define your feedforward pass
        x = self.fc(x)
        # squeeze -1 to 1 
        self.decoded = nn.tanh(x) #


class VAE(nn.Module):
    def __init__(self, airfoil_dim, latent_dim):
        super(VAE, self).__init__()
        self.enc = Encoder(airfoil_dim, latent_dim)
        self.dec = Decoder(latent_dim, airfoil_dim)
    
    def forward(self, x):
        # define your feedforward pass
        mean, self.enc.

    def decode(self, z):
        # given random noise z, generate airfoils
        return self.dec(z)

