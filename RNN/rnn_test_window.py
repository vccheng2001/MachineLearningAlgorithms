'''
rnn_test_window.py

After training is done, this program takes an input test file and 
and runs it against the trained model (trained_<apnea-type>_model).
Performs a sliding window (window size: <timesteps>) over the test file
and outputs a prediction file.

Output file: predictions_<apnea_type>_window.txt

params: <apnea_type>
        <timesteps/window_size> (same by default)
        <threshold>
Example: python3 rnn_test_window.py osa 160 0.9
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

# parameters 
(program, apnea_type, timesteps, threshold) = sys.argv
print(apnea_type)
test_path = "test_" + apnea_type + '/'
batch_size = 64

# Takes input test vector and runs it against trained LSTM model 
def main():
    # load saved model 
    model = keras.models.load_model(f"trained_{apnea_type}_model")
    # load input test vector 
    testX = load_test_dataset()
    # predict using sliding window 
    predictions = model.predict(testX, batch_size, verbose=1)
    # number of samples generated using sliding window 
    num_predictions = predictions.shape[0]
    # 1 if predict apnea, 0 otherwise 
    flags = np.zeros((num_predictions, 1))

    for i in range(num_predictions):
        p = predictions[i]
        # flag apnea (1) if positive prediction >= threshold, else 0
        flags[i] = 1 if p[1] >= threshold else 0
    predictions = np.hstack((predictions, flags))
    # Save to predictions file 
    np.savetxt(f'predictions_{apnea_type}_window.txt', predictions, delimiter=" ", fmt='%10f')


# create test X matrix 
def load_files_test(X):
    files = os.listdir(test_path)
    for file in files:
        print(f'Currently processing test file : {file}')
        # !! delete <usecols> if only 1 column
        arr = np.loadtxt(test_path + file,delimiter="\t", dtype=np.float64, usecols=[0])
        # create sliding window and add to test X 
        arr = window(arr, int(timesteps), 1, True)
        X = np.vstack((X,arr))
    print(f'Sliding window X shape: {X.shape}')
    return X

# creates sliding window 
def window(arr, window_size = int(timesteps), o = 1, copy = True):
    sh = (arr.size - window_size + 1, window_size)
    st = arr.strides * 2
    view = np.lib.stride_tricks.as_strided(arr, strides = st, shape = sh)[0::o]
    if copy: return view.copy()
    else: return view

# load input test vector as matrix 
def load_test_dataset():
    testX = np.array([], dtype=np.float64).reshape(0,int(timesteps))
    testX = load_files_test(testX)
    testX = np.expand_dims(testX, axis=2)
    return testX


if __name__ == "__main__":
    main()