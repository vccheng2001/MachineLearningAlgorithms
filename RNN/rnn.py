# lstm model
import os 
import numpy as np
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot

# lstm doesnt work well with > 400 timesteps
#rolling mean every 10 observations, choose every 10 

output_labels= ["positive", "negative"]

def load_files(label, group, X, y):
    path = group + '/' +label + '/'
    files = os.listdir(path)
    # Load (x, y))
    for file in files:
        # Load each x sample 
        # print(file)
        vec = np.loadtxt(path + file,delimiter="\n", dtype=np.float64)
        X = np.vstack((X,vec))
        # Append output class to y vector 
        label_num = 1 if label == "positive" else 0
        y = np.hstack((y,label_num))
    return X, y


def load_dataset():
    # Load Train Data 
    timesteps = 120
    group = "train"
    trainy = np.array([],dtype=np.int64)
    trainX = np.array([], dtype=np.float64).reshape(0,timesteps)
    # Load train files 
    for label in output_labels:
        trainX, trainy = load_files(label, group, trainX, trainy)
    trainX = np.expand_dims(trainX, axis=2)
   
    #Load Test data 
    group = "test"
    testy = np.array([],dtype=np.int64)
    testX = np.array([], dtype=np.float64).reshape(0,timesteps)
    # Load train files 
    for label in output_labels:
        testX, testy = load_files(label, group, testX, testy)
    testX = np.expand_dims(testX, axis=2)

    print('testx shape ', testX.shape)
    print("testy ",testy)
    
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)

    # print("trainy: ", trainy.shape)
    # print("trainX: ", trainX.shape)
    # print("testy: ", testy.shape)
    # print("testX: ", testX.shape)
    

    return trainX, trainy, testX, testy 


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
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
    # evaluate model
    # _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    # return accuracy
    return (model, batch_size)

# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def make_predictions():
    # load data
    trainX, trainy, testX, testy = load_dataset()
    # repeat experiment
    model, batch_size = evaluate_model(trainX, trainy, testX, testy)
    predictions = model.predict(testX, batch_size, verbose=1)
    print(predictions)

# run an experiment
def run_experiment(repeats=1):
    # load data
    trainX, trainy, testX, testy = load_dataset()
    # repeat experiment
    scores = list()
    for r in range(repeats):
        model, batch_size = evaluate_model(trainX, trainy, testX, testy)
        _, accuracy = model.evaluate(testX, testy, batch_size, verbose=0)
        score = accuracy * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)

# run the experiment
# run_experiment()
make_predictions()