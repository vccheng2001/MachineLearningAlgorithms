import numpy as np
import sys
import math 
import csv

# 1-hidden layer neural network for character identification

################################################################
#            STOCHASTIC GRADIENT DESCENT FOR NN                #
################################################################

# Perform Stochastic Gradient Descent to find optimal parameters alpha, Beta 
def SGD(alpha, Beta, X, y, yhat, dims, num_epochs, learning_rate, vX, vy, vyhat, vdims, metrics_out):
    m = open(metrics_out, "w")
    (M, D, K, N) = dims   # Train dims 
    (vM,vD,vK,vN) = vdims # Validation dims
    cross_entropy, vcross_entropy = {}, {}
    # For given number of epochs
    for epoch in range(num_epochs):
        # For current training example
        CE, vCE  = 0.0, 0.0
        avg_CE, vavg_CE = 0.0, 0.0
        for i in range(N):
            # ith training example 
            xi, yi, yhat_i = X[i].T, y[i].T, yhat[i].T 
            # Perform NN_forward, store obj = (x, a, b, z, yhat, J)
            obj = NN_forward(xi, yi, alpha, Beta)
            (xi, a, z, b, yhat_i, J) = obj # prediction before update
            yhat[i] = yhat_i.T  # Update yhat vector 
            # Backpropagation returns weight gradients
            (g_alpha, g_Beta)= NN_backward(xi,yi,alpha,Beta,obj)
            # Update weights 
            alpha = alpha - (learning_rate * (g_alpha))
            Beta = Beta - (learning_rate * (g_Beta))
        # Evaluate cross entropy 
        for j in range(N):
            xj, yj = X[j].T, y[j].T
            # new predicted labels using updated weights
            newyhat_j = NN_forward(xj, yj, alpha, Beta)[4] 
            for k in range(K): 
                CE += (yj[k][0] * math.log(newyhat_j[k][0]))
        # Average cross entropy 
        avg_CE = (-1/N) * CE
        cross_entropy[epoch+1] = avg_CE
        m.write("epoch=%s cross_entropy(train) : %s\n" % (str(epoch+1), str(avg_CE)))

        # Eval Cross entropy for validation  
        for j in range(vN): 
            vxj, vyj = vX[j].T, vy[j].T
            vnewyhat_j = NN_forward(vxj, vyj, alpha, Beta)[4] # predicted
            for k in range(K):
                vCE += (vyj[k][0] * math.log(vnewyhat_j[k][0])) 
        # Average cross entropy 
        vavg_CE = (-1/vN) * vCE
        vcross_entropy[epoch+1] = vavg_CE
        m.write("epoch=%s cross_entropy(validation) : %s\n" % (str(epoch+1), str(vavg_CE)))
    m.close()
    return (alpha, Beta)

################################################################
#                NEURAL NET: BACKPROPAGATION                   #
################################################################

# α and gα are D × (M +1), β and gβ are K × (D + 1)
# Neural Net: Backpropagation
def NN_backward(xi, yi, alpha, Beta, obj):
    # Put object into scope 
    (xi, a, z, b, yhat_i, J) = obj
    gJ = 1                                             # dl/dl
    gyhat = cross_entropy_backward(yi, yhat_i, J, gJ)  # dl/dyhat 
    gb = softmax_backward(b, yhat_i, gyhat)            # dl/db
    gBeta, gz = linear_backward(z, b, gb, Beta)        # dl/dBeta, dl/dz
    ga = sigmoid_backward(a, z, gz)                    # dl/dalpha, dl/dx
    galpha, gx = linear_backward(xi, a, ga, alpha)
    # Gradient of alpha, beta (used to update weights in SGD)
    return (galpha, gBeta)

# get dl/dyhat
def cross_entropy_backward(yi, yhat_i, J, gJ):
    gyhat = gJ * np.multiply(yi, (-1/yhat_i))
    gyhat = gyhat.T 
    return gyhat 

# get dl/db = (dl/dyhat)*(dyhat/db) = gyhat * (dyhat/db)
def softmax_backward(b, yhat_i, gyhat):
    K = b.size 
    mat = np.eye(K) # Identity matrix 
    for i in range(K): 
        for j in range(K):  # Get derivative of softmax
            if i == j:      # If diagonal, S(bi) * (1-S(bi))
                mat[i, i] = (yhat_i[i]) * (1-yhat_i[i])
            else:           # Else,       -S(bi) * S(bj) 
                mat[i, j] = -yhat_i[i] * yhat_i[j]
    gb = np.matmul(gyhat,mat) # gyhat is 1x1K row vector, mat = KxK matrix
    return gb

# dl/dBeta = dl/db*db/dBeta = (gb)*(db/dBeta) 
# dl/dz = (gb)*(db/dz) = gb*beta
def linear_backward(input, w, gw, wMatrix):
    zT = input.T
    gwT = gw.T 
    gweight = np.dot(gwT, zT) # db/dBeta: gBeta should be K*(D+1)
    ginput = np.dot(gw, wMatrix) # db/dz: Beta matrix 
    # delete first element from gz (bias)
    ginput = ginput[:,1:]
    return (gweight, ginput)


# dl/da = (dl/dz)*(dz/da) = gz*(dz/da). 
def sigmoid_backward(a,z,gz):
    D = a.shape[0]
    dz_da = np.eye(D)
    # remove first row (bias element) of col vector z
    z = np.delete(z, 0, axis = 0) 
    z_remove_dim = np.squeeze(z)
    # Derivative of sigmoid: z*(1-z), where z = sigmoid(a)
    dz_da[range(D), range(D)] = (z_remove_dim*(1-z_remove_dim))
    ga = np.dot(gz, dz_da)
    return ga

################################################################
#                NEURAL NET: FORWARD PASS                      #
################################################################


# Inputs: training example (xi, yi), weights alpha, Beta
def NN_forward(xi, yi, alpha, Beta):
    a = linear_forward(xi, alpha)          # a: lc of x, weight matrix alpha (D*1)
    z = sigmoid_forward(a)                 # z = sigmoid(a) (D*1)
    b = linear_forward(z, Beta)            # b: lc of z, weight matrix Beta (K*1)
    yhat_i = softmax_forward(b)            # yhat: softmax of b (K*1)
    J = cross_entropy_forward(yi, yhat_i)  # cross_entropy: -(sum(yi*log(yhat_i)))
    obj = (xi, a, z, b, yhat_i, J)         # store intermediate objects
    return obj

# Perform linear combination of weights with inputs 
def linear_forward(inputs, weights):
    a = np.dot(weights, inputs)
    return a

# z = sigmoid(a), return z with bias (first element: 1)
def sigmoid_forward(a):
    z = 1 / (1 + np.exp(-a))
    z_with_bias = np.concatenate(([[1]],z))
    return z_with_bias 

# yhat = softmax(b)
def softmax_forward(b):
    yhat_i = np.exp(b) / np.sum(np.exp(b), axis=0)
    return yhat_i

# Calculates cross entropy given true label y_i, predicted yhat_i for ith training example
def cross_entropy_forward(yi, yhat_i):
    J = -1 * np.sum(yi*np.log(yhat_i))
    return J

################################################################
#              NEURAL NET: PRE-PROCESSING                      #
################################################################

def main():
    # Parse command line args 
    (program, trn_in, valid_in, trn_out, valid_out, metrics_out, num_epochs, hidden_units, 
                                                     init_flag, learning_rate) = sys.argv
    # Setup train 
    (tX, ty, tyhat, tdims) = setup(trn_in, hidden_units)
    # Setup valid
    (vX, vy, vyhat, vdims) = setup(valid_in, hidden_units)
    # Initialize weights for training (if init_flag == 1, RANDOM, else if 2, ZERO) 
    (talpha, tBeta) = init_weights(init_flag, tdims)
    # Run SGD with train data 
    (talpha_new, tBeta_new) = SGD(talpha, tBeta, tX, ty, tyhat, tdims, int(num_epochs), 
                               float(learning_rate), vX, vy, vyhat, vdims, metrics_out) 
    m = open(metrics_out, 'a')
    # Train predictions 
    terror_rate = make_predictions(trn_out, tdims, tX, ty, talpha_new, tBeta_new)
    m.write('error(train): %s\n' % terror_rate)
    # Validation predictions 
    verror_rate = make_predictions(valid_out, vdims, vX, vy, talpha_new, tBeta_new)
    m.write('error(validation): %s\n' % verror_rate)
    m.close()

# Retrieve dimensions, build X, y, yhat from file
# D: number of hidden units, K: number of classes (0-9), M: 128 features, N: number of samples
def setup(in_file, hidden_units):
    (M, N) = file_len(in_file)
    (M, D, K, N) = (M, int(hidden_units), 10, N)
    # Get dimensions, X: N*(M+1), alpha: D*(M+1) Beta: K*(D+1) 
    dims = (M, D, K, N)
    # Build X 
    X = build_X(in_file)
    # Build true labels y
    y = build_y(in_file, dims)
    # Build predicted labels yhat
    yhat = build_yhat(dims)
    return (X, y, yhat, dims)

# Make predictions for train and validation 
def make_predictions(out_file, dims, X, y, talpha_new, tBeta_new):
    errors = 0
    (M, D, K, N) = dims
    # Initialize final prediction vector 
    final_yhat = np.zeros((N, K)) # N samples, each is a K*1 vector
    final_yhat = np.expand_dims(final_yhat, axis=1)
    # Using optimized talpha, tBeta, get final yhat (predictions)
    for i in range(N):
        final_obj = NN_forward(X[i].T, y[i].T, talpha_new, tBeta_new)
        final_yhat[i] = final_obj[4].T
    # Open prediction .labels outfile (either train/validation)
    pred_file = open(out_file, 'w')
    # Write predictions for each train sample in output predicted file 
    for i in range(N): 
        # Class that yields highest probability for sample i (argmax for yhat_i)
        predicted_class = np.argmax(final_yhat[i], axis=1)
        # True class (argmax for yi)
        true_class = np.argmax(y[i], axis=1)
        # If wrong prediction, increment errors
        if (predicted_class != true_class): errors += 1
        # Write prediction to .labels outfile 
        pred_file.write(str(predicted_class[0]) + "\n")
    # Error rate 
    error_rate = errors * (1/N)
    pred_file.close()
    return error_rate
    


# Builds X, a N*(M+1) matrix with first column being all 1s (bias term)
def build_X(file):
    X = np.genfromtxt(file, delimiter=',') # generate 2-D matrix from text file 
    X = np.delete(X, 0, axis = 1)          # delete first column (y-label)
    X_bias = np.ones((X.shape[0], 1))      # Bias column
    X = np.hstack((X_bias, X))             # Add bias column of all 1s
    X = np.expand_dims(X, axis=1)          
    return X

# Builds true y, where each y(i) is a one-hot vector where if k is true label, y_i[k] = 1
def build_y(file, dims): # for each y(i), K*1
    (M, D, K, N) = dims
    y = np.zeros((N, K)) # Init to 0s
    # Labels are first column of text file 
    true_labels = np.genfromtxt(file, delimiter = ",", usecols=(0)) 
    for i in range(N):
        # for given sample, get true class
        true_label = int(true_labels[i]) 
        # create one hot vector representing true label
        y[i, true_label] = 1 
    y = np.expand_dims(y, axis=1)
    return y

# Builds predicted y, where each yhat(i) is a K-length vector of probability distributions
def build_yhat(dims): 
    (M, D, K, N) = dims
    yhat = np.zeros((N, K)) # Init to 0, 10 samples, each is a K-length vector 
    yhat = np.expand_dims(yhat, axis=1)
    return yhat

# Initializes alpha, beta matrices (either RANDOMIZE or ZERO)
def init_weights(init_flag, dims):
    (M, D, K, N) = dims
    if init_flag == '1':
        # Weights random from -0.1 to 0.1, bias terms 0 
        alpha = np.random.uniform(-0.1, 0.1, (D,M))
        alpha_bias = np.zeros((alpha.shape[0], 1))
        alpha = np.hstack((alpha_bias, alpha))
        Beta = np.random.uniform(-0.1, 0.1, (K,D))
        Beta_bias = np.zeros((Beta.shape[0], 1))
        Beta = np.hstack((Beta_bias, Beta))
    elif init_flag == '2':
        # Zero weights and bias 
        alpha = np.zeros((D, M+1))
        Beta = np.zeros((K, D+1))
    return (alpha, Beta)

# Returns number of features, number of samples given input file
# Each row of file: one training example
# First col of file: y labels. Remaining cols: features 
def file_len(fname):
    ff = open(fname, 'r')
    reader = csv.reader(ff, delimiter=',')
    M = len(next(reader)) - 1
    ff.close()
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    f.close()
    N = i + 1
    # Returns number of features, number of samples 
    return (M, N)

if __name__ == "__main__":
    main()