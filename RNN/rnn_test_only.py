'''
rnn_test_only.py

This program makes predictions on files in the 
test directory using the pre-trained model. 
** must train first 

params: <apnea_type>, <timesteps>, <threshold>
Example: python3 rnn_train_and_test.py osa 160 0.9
'''

import os, sys
import numpy as np
from numpy import mean, std, dstack 
from pandas import read_csv

# Keras LSTM model 
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical

# Graphing 
from matplotlib import pyplot

# parameters 
(program, apnea_type, timesteps, threshold) = sys.argv
test_path = "test_" + apnea_type + '/'
batch_size = 64
labels = {"positive/":1, "negative/":0}


# tests on positive/negative sequences in test files
def main():
    # load saved model 
    model = keras.models.load_model(f"trained_{apnea_type}_model")
    # load input test vector 
    testX, actual = load_test_dataset()
    # predictions based on trained model 
    predictions = model.predict(testX, batch_size)
    # save predictions to file 
    output_predictions(predictions, np.asarray(actual))
   
# saves predictions 
def output_predictions(predictions,actual):
    # number of test files in test dir
    num_predictions = predictions.shape[0] 
    # 1 if > threshold, 0 otherwise 
    flags = np.zeros((num_predictions, 1))
    for i in range(num_predictions):
        p = predictions[i]
        # flag apnea (1) if positive prediction >= threshold, else 0
        flags[i] = 1 if p[1] >= float(threshold) else 0
    
    # Add column for flagged (predicted) values
    predictions = np.hstack((predictions, flags))

    # Add column for actual values
    actual = np.expand_dims(actual, axis=1)
    predictions = np.hstack((predictions, actual))

    # save predictions
    np.savetxt(f'predictions_{apnea_type}.txt', predictions, delimiter=' ',fmt='%10f',header="Negative,Positive,Prediction,Actual")
    print(predictions)

# Create test X matrix, actual values
def load_files_test(actual, label, X):
    path=test_path+label
    files = os.listdir(path)
    for file in files:
        print('Currently processing test file:', file)
        arr = np.loadtxt(path + file,delimiter="\n", dtype=np.float64)
        # Add as row to x matrix
        X = np.vstack((X, arr))
        # Build actual values (positive/negative)
        actual.append(labels[label])
    return X

# load input test vector as matrix 
def load_test_dataset():
    actual = list()
    testX = np.array([], dtype=np.float64).reshape(0,int(timesteps))
    # Load test files 
    for label in labels:
        testX = load_files_test(actual, label, testX)
    testX = np.expand_dims(testX, axis=2)
    return testX, actual


if __name__ == "__main__":
    main()