
import numpy as np 
import math

# Gradient Descent, GeLU Activation

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def gelu(x):
  return x * sigmoid(1.702*x)

# gradient 
def grad(x):
    return sigmoid(1.702*x) * (1+1.702*x*(1-sigmoid(1.702*x)))

# Set up variables and run GD
def main():
    lr = 0.1                                    # learning rate 
    x = {0:-3,1:0,2:0,3:0}                      # store x[i] 
    gelu_dict = {0:0, 1:0, 2:0, 3:0}            # store gelu(x[i]) 
    num_iters = 3
  
    for i in range(num_iters): 
        x[i+1] = x[i] - lr * (grad(x[i]))       # update x[i+1] 
        gelu_dict[i+1] = gelu(x[i+1])           # get gelu of new x[i+1]


if 'name == __main__':
    main()
