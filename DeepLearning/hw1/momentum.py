
import numpy as np 
import math

# Gradient Descent with Momentum, GeLU Activation

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def gelu(x):
  return x * sigmoid(1.702*x)

def grad(x):
    return sigmoid(1.702*x) * (1+1.702*x*(1-sigmoid(1.702*x)))
    # l = sigmoid(1.702*x)
    # r = 1.702*x*(sigmoid(1.702*x) * (1-sigmoid(1.702*x)))
    # return l + r

# Set up variables and run perceptron 
def main():
    B = 0.9
    lr = 1
    x = {0:-3,1:0,2:0,3:0}
    v = {0:grad(x[0]), 0:0, 0:0, 0:0}
    gelu_dict = {0:0, 1:0, 2:0, 3:0}
    num_iters = 3
    print(f"GD with Momentum")
    print(f"lr: {lr}, x_0: {x[0]}, num_iters: {num_iters}, B: {B}")
    for i in range(3): 
        v[i+1] = (B*v[i]) + ((1-B)*grad(x[i]))
        # update x[i+1]
        x[i+1] = x[i] - (lr * v[i+1])
        print(f"x_{i+1}:        {x[i+1]}")
        # get gelu of new x[i+1]
        gelu_dict[i+1] = gelu(x[i+1])
        print(f"GeLU(x_{i+1}): ", gelu_dict[i+1])


if 'name == __main__':
    main()