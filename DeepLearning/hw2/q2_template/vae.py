import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        # build your model here
        # your model should output a predicted mean and a predicted std of the encoding
        # both should be of dim (batch_size, latent_dim)
    
        # batch_size = 16, input_dim = 16, latent_dim = 16
        '''
        UIUC Airfoil Coordinates Database includes coordinates for nearly 1,600 airfoils.
        Since number of points in each sample differ, we first process all airfoils to
        have 200 points and share the same x coordinates via spline interpo- lation.
        Also, all y coordinates are rescaled to [-1,1]. The processed airfoils are 
        shown in Fig. 7a. Therefore, only y coordinates of each airfoil is used to train and test generative models.
        '''
        self.fc1 = nn.Linear(input_dim, 100)
        self.relu = nn.ReLU()
        self.fc21 = nn.Linear(100, latent_dim)  # mu layer
        self.fc22 = nn.Linear(100, latent_dim)  # logvariance layer

    def forward(self, x):
        # define your feedforward pass
        h1 = self.fc1(x)
        mu = self.relu(self.fc21(h1))
        logvar = self.relu(self.fc22(h1))
        print(mu.shape, logvar.shape)
        # both mu, logvar: 16 x 16
        return (mu, logvar)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        # build your model here
        # your output should be of dim (batch_size, output_dim)
        # you can use tanh() as the activation for the last layer
        # since y coord of airfoils range from -1 to 1
        self.fc3 = nn.Linear(latent_dim, 100)
        # from hidden 400 to 784 outputs
        self.fc4 = nn.Linear(100, output_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()


    def forward(self, x):
        h3 = self.relu(self.fc3(x))
        return self.tanh(self.fc4(h3))

class VAE(nn.Module):
    def __init__(self, airfoil_dim, latent_dim):
        super(VAE, self).__init__()
        self.airfoil_dim = airfoil_dim  
        self.latent_dim = latent_dim
        self.encoder = Encoder(airfoil_dim, latent_dim)
        self.decoder = Decoder(latent_dim, airfoil_dim)
    
    def forward(self, x):
        (mu, logvar) = self.encoder(x.view(-1, self.airfoil_dim))
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        print(f"z: {z}")
        recon_batch = self.decoder(z)
        # 16 x 200
        print(f"recon_batch: {recon_batch.shape}")
        return (recon_batch, mu, logvar)
        # self.z = self.reparameterize(self.mu, self.logvar)

    # def decode(self, z):
    #     # given random noise z, generate airfoils
    #     return self.decoder(z), self.mu, self.logvar 

