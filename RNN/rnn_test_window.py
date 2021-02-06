# after training, load saved model
# takes in one file, sliding window to make Xtest
import tensorflow as tf
from tensorflow import keras
import os 
import numpy as np
import pandas as pd
from numpy import mean, std, dstack
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical

timesteps = 160
threshold = 0.9
batch_size = 64

apnea_type = "osa"
test_group = "test_" + apnea_type
def test():
    model = keras.models.load_model('trained_'+apnea_type+'_model')

    testX, actual = load_test_dataset()
    predictions = model.predict(testX, batch_size, verbose=1)
    
    num_rows = predictions.shape[0]
    flagged_apnea_col = np.zeros((num_rows, 1))
    for i in range(num_rows):
        row = predictions[i]
        if row[1] >= threshold:
            flagged_apnea_col[i] = 1
    predictions = np.hstack((predictions, flagged_apnea_col))
    # predictions = np.hstack((predictions, actual))
    np.savetxt('predictions_'+apnea_type+'_window.txt', predictions, delimiter=' ', fmt='%10f') #"Negative,Positive,Predict")

    print(predictions)


def load_files_test(X):
    path = test_group+'/'
    files = os.listdir(path)
    for file in files:
        print('Currently processing test file :', file)

        # Use this for reading all columns of test file 
        #arr = np.loadtxt(path + file,delimiter="\n", dtype=np.float64)
        
        # Use this for only reading first column 
        arr = np.loadtxt(path + file,delimiter="\t", dtype=np.float64,usecols=[0])
       
        # actual = np.loadtxt(path + file,delimiter="\t", dtype=np.float64,usecols=[1])
        actual = 0
        arr = window(arr, timesteps, 1, True)
        X = np.vstack((X,arr))
    # print(X.shape, X)
    return X, actual

def window(arr, w = timesteps, o = 1, copy = True):
    sh = (arr.size - w + 1, w)
    st = arr.strides * 2
    view = np.lib.stride_tricks.as_strided(arr, strides = st, shape = sh)[0::o]
    if copy: return view.copy()
    else: return view

def load_test_dataset():
    testX = np.array([], dtype=np.float64).reshape(0,timesteps)
    # Load train files 
    testX, actual = load_files_test(testX)
    testX = np.expand_dims(testX, axis=2)
    print("test X ", testX.shape)
    return testX, actual

test()