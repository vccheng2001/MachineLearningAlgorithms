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

timesteps = 120

def test():
    batch_size = 64
    model = keras.models.load_model('saved_model')

    testX = load_test_dataset()
    predictions = model.predict(testX, batch_size, verbose=1)
    print(predictions)


def load_files_test( X):
    path = "test/"
    files = os.listdir(path)
    for file in files:
        print('Currently processing test file :', file)
        arr = np.loadtxt(path + file,delimiter="\n", dtype=np.float64)
        arr = window(arr, timesteps, 1, True)
        X = np.vstack((X,arr))
    return X

def window(arr, w = timesteps, o = 1, copy = True):
    sh = (arr.size - w + 1, w)
    st = arr.strides * 2
    view = np.lib.stride_tricks.as_strided(arr, strides = st, shape = sh)[0::o]
    if copy: return view.copy()
    else: return view

def load_test_dataset():
    testX = np.array([], dtype=np.float64).reshape(0,timesteps)
    # Load train files 
    testX = load_files_test(testX)
    testX = np.expand_dims(testX, axis=2)
    print("test X ", testX.shape)
    return testX

test()