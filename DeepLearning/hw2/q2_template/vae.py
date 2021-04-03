import torch
import torch.nn as nn

HIDDEN_DIM = 64

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        # build your model here
        # your model should output a predicted mean and a predicted std of the encoding
        # both should be of dim (batch_size, latent_dim)
    
        # batch_size = 16, input_dim = 200, latent_dim = 16
        '''
        UIUC Airfoil Coordinates Database includes coordinates for nearly 1,600 airfoils.
        Since number of points in each sample differ, we first process all airfoils to
        have 200 points and share the same x coordinates via spline interpo- lation.
        Also, all y coordinates are rescaled to [-1,1]. The processed airfoils are 
        shown in Fig. 7a. Therefore, only y coordinates of each airfoil is used to train and test generative models.
        '''

        # input (airfoil)  dim -> hidden_dim -> latent_dim
        self.fc1 = nn.Linear(input_dim, HIDDEN_DIM)
        self.fc21 = nn.Linear(HIDDEN_DIM, latent_dim)  # mu layer
        self.fc22 = nn.Linear(HIDDEN_DIM, latent_dim)  # logvariance layer
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.fc1(x)
        mu = self.relu(self.fc21(h1))
        logvar = self.relu(self.fc22(h1))
        # mu: ean matrix (batch_size * latent_dim)
        # logvar variance matrix (batch_size * latent_dim)
        return (mu, logvar)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        # build your model here
        # your output should be of dim (batch_size, output_dim)
        # you can use tanh() as the activation for the last layer
        # since y coord of airfoils range from -1 to 1

        # latent_dim-> hidden_dim ->output (airfoiL) dim
        self.fc3 = nn.Linear(latent_dim, HIDDEN_DIM)
        self.fc4 = nn.Linear(HIDDEN_DIM, output_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        h3 = self.relu(self.fc3(x))
        return self.tanh(self.fc4(h3))

class VAE(nn.Module):
    def __init__(self, airfoil_dim, latent_dim):
        super(VAE, self).__init__()
        # 200->....->16->....->200
        self.airfoil_dim = airfoil_dim  
        self.latent_dim = latent_dim
        # init encoder, decoder 
        self.encoder = Encoder(airfoil_dim, latent_dim)
        self.decoder = Decoder(latent_dim, airfoil_dim)

        self.mu = None      # mean matrix (batch_size * latent_dim)
        self.logvar = None  # variance matrix (batch_size * latent_dim)

    def reparameterize(self, mu, logvar):
        # for each training sample, take current learned mu, stddev
        # and draw a random sample from that distribution
        
        # multiply log variance with 0.5, then in-place exponent
        # to get the standard deviation
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # encode 
        (self.mu, self.logvar) = self.encoder(x.view(-1, self.airfoil_dim))
        # sample
        z = self.reparameterize(self.mu, self.logvar)
        # decode 
        return self.decode(z), self.mu, self.logvar 

    def decode(self, z):
        # decode given noise 
        return self.decoder(z)

