import tensorflow as tf
from tensorflow import keras
# lstm model
import os 
import numpy as np
from numpy import mean, std, dstack
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical


def test():
    batch_size = 64
    model = keras.models.load_model('saved_model')

    testX = load_test_dataset()
    predictions = model.predict(testX, batch_size, verbose=1)
    print(predictions)


def load_files_test(label, X):
    path = "test" + '/' +label + '/'
    files = os.listdir(path)
    for file in files:
        vec = np.loadtxt(path + file,delimiter="\n", dtype=np.float64)
        X = np.vstack((X,vec))
    return X

def load_test_dataset():
    timesteps = 120 
    testX = np.array([], dtype=np.float64).reshape(0,timesteps)
    # Load train files 
    for label in ["positive", "negative"]:
        testX = load_files_test(label, testX)
    testX = np.expand_dims(testX, axis=2)
    print("test X ", testX.shape)
    return testX
test()