'''
Utility functions 
Free from to add functions if needed
'''

import torch
import matplotlib.pyplot as plt



def plot_airfoils(airfoil_x, airfoil_y):
    '''
    plot airfoils: no need to modify 
    '''
    idx = 0
    fig, ax = plt.subplots(nrows=4, ncols=4)
    for row in ax:
        for col in row:
            col.scatter(airfoil_x, airfoil_y[idx, :], s=0.6, c='black')
            col.axis('off')
            col.axis('equal')
            idx += 1
    plt.show()


def label_real(size):
    data = torch.ones(size, 1)
    return data
# to create fake labels (0s)
def label_fake(size):
    data = torch.zeros(size, 1)
    return data
# function to create the noise vector
def create_noise(batch_size, latent_dim):
    return torch.randn(batch_size, latent_dim)