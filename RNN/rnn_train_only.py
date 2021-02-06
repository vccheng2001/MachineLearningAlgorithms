# Does train only and saves model into saved_model
import os 
import numpy as np
from numpy import mean, std, dstack 

from pandas import read_csv

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical

from matplotlib import pyplot

# number of timesteps (10 sec before apnea, 5 after)
# 15 * 8 samples/sec = 120 samples
timesteps = 160
apnea_type = "osa"
train_group = "train_" + apnea_type 

def load_files_train(label, X, y):
    path = train_group + '/' +label + '/'
    files = os.listdir(path)
    for file in files:
        vec = np.loadtxt(path + file,delimiter="\n", dtype=np.float64)
        X = np.vstack((X,vec))
        # Append output class to y vector 
        label_num = 1 if label == "positive" else 0
        y = np.hstack((y,label_num))
    return X, y

def load_train_dataset():
    # Load Train Data 
    trainy = np.array([],dtype=np.int64)
    trainX = np.array([], dtype=np.float64).reshape(0,timesteps)
    # Load train files for both labels
    for label in ["positive", "negative"]:
        trainX, trainy = load_files_train(label, trainX, trainy)
    trainX = np.expand_dims(trainX, axis=2)
    trainy = to_categorical(trainy)
    return trainX, trainy


# fit and evaluate a model
def train_model(trainX, trainy):
    verbose, epochs, batch_size = 0, 15, 64
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

def train():
    # load train data, train 
    trainX, trainy = load_train_dataset()
    model = train_model(trainX, trainy)
    model.save('trained_'+apnea_type+'_model')

train()