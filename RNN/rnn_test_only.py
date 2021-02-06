import tensorflow as tf
from tensorflow import keras
# lstm model
import os 
import numpy as np
import pandas as pd
from numpy import mean, std, dstack
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical

timesteps = 120
threshold = 0.5
batch_size = 64
output_labels= ["positive", "negative"]

apnea_type = "osa"
test_group = "test_" + apnea_type 
def test():
    model = keras.models.load_model('trained_'+apnea_type+'_model')
    testX, actual = load_test_dataset()
    probs = model.predict(testX, batch_size, verbose=1)
    output_predictions(probs,actual)
   

def output_predictions(probs,actual):
    N = probs.shape[0]
    pred = np.zeros((N, 1))
    for i in range(N):
        row = probs[i]
        if row[1] >= threshold:
            pred[i] = 1
    output = np.hstack((probs, pred))

    # Actual
    actual = np.asarray(actual)
    actual = np.expand_dims(actual, axis=1)

    # Add pred, actual columns
    pred_actual = np.hstack((pred,actual)) 
    output = np.hstack((probs, pred_actual))
    np.savetxt('predictions_'+apnea_type+'.txt', output, delimiter=' ',fmt='%10f',header="Negative,Positive,Prediction,Actual")
    print(output)



def load_files_test(actual, label, X):
    path = test_group + '/' +label + "/"
    files = os.listdir(path)
    for file in files:
        print('Currently processing test file:', file)
        vec = np.loadtxt(path + file,delimiter="\n", dtype=np.float64)
        X = np.vstack((X,vec))
        if 'positive' in file:
            actual.append(1)
        else:
            actual.append(0)
    return X

def load_test_dataset():
    actual= []
    testX = np.array([], dtype=np.float64).reshape(0,timesteps)
    # Load train files 
    for label in output_labels:
        testX = load_files_test(actual, label, testX)
    testX = np.expand_dims(testX, axis=2)
    print("test X ", testX.shape)
    return testX,actual

test()