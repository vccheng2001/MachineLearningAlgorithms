# lstm model
import os 
import numpy as np
from numpy import mean, std, dstack 
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot


def load_files_train(label, X, y):
    path = "train" + '/' +label + '/'
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
    timesteps = 120
    trainy = np.array([],dtype=np.int64)
    trainX = np.array([], dtype=np.float64).reshape(0,timesteps)
    # Load train files 
    for label in ["positive", "negative"]:
        trainX, trainy = load_files_train(label, trainX, trainy)
    trainX = np.expand_dims(trainX, axis=2)
    trainy = to_categorical(trainy)
    return trainX, trainy


# fit and evaluate a model
def train_model(trainX, trainy):
    verbose, epochs, batch_size = 0, 15, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model

def train():
    # load train data, train 
    trainX, trainy = load_train_dataset()
    model = train_model(trainX, trainy)
    model.save('saved_model')

train()