'''
rnn_train_only.py

This program reads the train files into X, y matrices,
then trains/saves an RNN model to the file trained_<apnea-type>_model.
'''

import os, sys
import numpy as np
from numpy import mean, std, dstack 
from pandas import read_csv

# Keras LSTM model 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical

# Graphing 
from matplotlib import pyplot

# parameters
(program, apnea_type, timesteps) = sys.argv
labels = {"positive/":1, "negative/":0}
train_group = f"train_{apnea_type}/"
epochs, batch_size = 15, 64

# fit and evaluate rnn-lstm model
def train_model(trainX, trainy):
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    # Add one layer at a time 
    model = Sequential()
    # 100 units in output 
    model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
    # drop 50% of input units 
    model.add(Dropout(0.5))
    # dense neural net layer, relu(z) = max(0,z) output = activation(dot(input, kernel)
    model.add(Dense(100, activation='relu'))
    # Softmax: n_outputs in output (1)
    model.add(Dense(n_outputs, activation='softmax'))
    # Binary 0-1 loss, use SGD 
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model

# load train files for positive and negative sequences 
def load_train_dataset():
    # Load Train Data 
    trainy = np.array([],dtype=np.int64)
    trainX = np.array([], dtype=np.float64).reshape(0,int(timesteps))
    # Load train files for positive and negative sequences 
    for label in labels:
        trainX, trainy = load_files_train(label, trainX, trainy)
    trainX = np.expand_dims(trainX, axis=2)
    # convert y into a two-column probability distribution (-, +)
    trainy = to_categorical(trainy)
    return trainX, trainy

# Creates X, y train matrices 
def load_files_train(label, trainX, trainy):
    path = train_group+label # e.g. train/positive/
    files = os.listdir(path)
    # Load all N train files one sample at a time
    for file in files:
        sample= np.loadtxt(path + file,delimiter="\n", dtype=np.float64)
        trainX = np.vstack((trainX, sample))
        # Append binary 1 or 0 to y vector 
        binary_label = labels[label]
        trainy = np.hstack((trainy,binary_label))
    return trainX, trainy

def main():
    trainX, trainy = load_train_dataset()       # Load train data 
    model = train_model(trainX, trainy)         # Train model 
    model.save(f'trained_{apnea_type}_model')   # Save model


if __name__ == "__main__":
    main()