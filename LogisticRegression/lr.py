import numpy as np
import sys
import math 

#  Write a function that takes a single SGD step on thei-th training example.  
#  Such a function shouldtake as input the model parameters, the learning rate,
#   and the features and label for thei-th trainingexample. It should update the model parameters 
#   in place by taking one stochastic gradient step

# [ 0 0 0 0 0 0 010 01  1] , [1, 12, 14, 15 ,16..]
# # X[i] 
def dot(xi, theta):
    sum = 0
    for key, val in xi.items():
        sum += (1 * theta[key+1])
    sum += (1*theta[0]) # bias 
    return sum 
    

def SGD(X, y, theta, alpha, num_epochs, M, N):
    for epoch in range(num_epochs):
        for i in range(0, N):
            xi = X[i]
            yi = y[i]
            theta_transpose_x = dot(xi, theta)
            exp = math.exp(theta_transpose_x)
            grad = float(yi) - (exp / (1+exp))
            for key, val in xi.items():
                theta[key+1] = theta[key+1]  + ((alpha*val/N) * grad)
            theta[0] = theta[0] + ((alpha*1/N) * grad )#bias 
    return theta



def make_predictions(theta, X, y, N):
    predictions, errors = [], 0
    for i in range(N): 
        xi, yi = X[i], y[i]
        theta_transpose_x = dot(xi, theta)
        exp = math.exp(theta_transpose_x)
        p = (exp) / float(1+exp)
        predicted_label = 1 if (p >= 0.5) else 0
        predictions.append(predicted_label)
        if yi != predicted_label:
            errors += 1
    error_rate = float(errors / N)
    return (predictions, error_rate)

def main():
    (program, ftrn_in, fvalid_in, ftest_in, dict_in, trn_labels, test_labels, metrics_out, num_epochs) = sys.argv
    alpha = 0.1 
    files = [(ftrn_in, trn_labels), (ftest_in, test_labels)]

    # Keep track of train/test error 
    err_strings = []
    M = file_len(dict_in)
    # Train only
    # Setting up variables 
    (in_file, out_file) = files[0]
    theta = [0] * (M+1)
    y, X = [], {}
    y = build_y(y,in_file)
    X = build_X(X, in_file)
    N = len(X)
    params = SGD(X,y,theta,alpha, int(num_epochs), M, N)
    (predictions, error_rate) = make_predictions(params, X, y, N)
    # predictions 
    write_predictions(out_file, predictions)
    err_strings.append("error(train): " + str(error_rate))

    # Now test 
    # Setting up variables 
    (in_file, out_file) = files[1]
    test_y, test_X = [], {}
    test_y= build_y(test_y,in_file)
    test_X = build_X(test_X, in_file)
    N = len(test_X) 
    # predict, calculate error rate on test set 
    (predictions, error_rate) = make_predictions(params, test_X, test_y, N)
    write_predictions(out_file, predictions)
    err_strings.append("error(test): " + str(error_rate))
    print_errors(err_strings, metrics_out)

def print_errors(err_strings, metrics_out):
    m = open(metrics_out, "w")
    for s in err_strings:
        m.write(s + "\n")
    m.close()

def write_predictions(out_file, predictions): 
    pred_file = open(out_file, 'w')
    for p in predictions: 
        pred_file.write(str(p) + "\n")
    pred_file.close()

def build_y(y, fname):
    f = open(fname, 'r')
    for row in f:
        yi = float(row[0])
        y.append(yi)
    f.close()
    return y

def build_X(X, fname):
    f = open(fname, 'r')
    i = 0
    for row in f:
        X[i] = {} #one row/movie review
        vals = row.split('\t')[1:]
        for v in vals:
            dict_index = int(v.split(':')[0])
            X[i][dict_index] = 1 # make sure to add bias 
        i += 1
    f.close()
    return X

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    f.close()
    return i + 1

if __name__ == "__main__":
    main()