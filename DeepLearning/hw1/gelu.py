
import numpy as np 
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def gelu(x):
  return x * sigmoid(1.702*x)

# gradient 
def grad(x):
    return sigmoid(1.702*x) * (1+1.702*x*(1-sigmoid(1.702*x)))

# Set up variables and run GELU GD
def main():
    lr = 0.1                                # learning rate 
    x = {0:-3,1:0,2:0,3:0}                   # store x[i] 
    gelu_dict = {0:0, 1:0, 2:0, 3:0}        # store gelu(x[i]) 
    num_iters = 3
    print("****************************")
    print(f"Using lr: {lr}, x_0:{x[0]}, num_iters: {num_iters}")
    for i in range(num_iters): 
        x[i+1] = x[i] - lr * (grad(x[i]))   # update x[i+1] 
        print(f"x_{i+1}:       ", x[i+1])
        gelu_dict[i+1] = gelu(x[i+1])       # get gelu of new x[i+1]
        print(f"GeLU(x_{i+1}): ", gelu_dict[i+1])


if 'name == __main__':
    main()

# lr = 0.1
# x: {0: 0, 1: -0.05, 2: -0.09575013021851926, 3: -0.13763771842581057}
# gelu: {0: 0, 1: -0.02393689150943369, 2: -0.04398265469655064, 3: -0.06079478852684933}

# lr:  1
# x:  {0: 0, 1: -0.5, 2: -0.6207780880345858, 3: -0.676497308917168}
# gelu : {0: 0, 1: -0.1496115633936199, 2: -0.1601399925974373, 3: -0.16251748448720163}i

# x_0 is... -3
# lr:  0.1
# x:  {0: -3, 1: -2.997545167609435, 2: -2.995082708753491, 3: -2.9926125791343785}
# gelu : {0: 0, 1: -0.01813166529499688, 2: -0.018192396733478253, 3: -0.018253507384506158}

# x_0 is... -3
# v_0 is... 0
# lr:  1
# x:  {0: -3, 1: -2.997545167609435, 2: -2.9928733596019823, 3: -2.986191702877126}
# gelu : {0: 0, 1: -0.01813166529499688, 2: -0.01824704671202677, 3: -0.01841325299731993}

# GD with Momentum reaches a more optimal point/extremum (-0.0184 < -0.01825) in
# the same number of iterations 