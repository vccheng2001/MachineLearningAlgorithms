
import pandas as pd 
import os
import numpy as np
import csv
import shutil
import sys

(program, apnea_type, timesteps) = sys.argv
raw_path = "raw_" + apnea_type + '/'
train_path = "train_" + apnea_type + '/'
test_path = "test_" + apnea_type + '/'
labels = ["positive/", "negative/"]

# Preprocesses apnea files 
def main():
    init_dirs()
    for label in labels:
        setup_train_data(raw_path, label)
    for label in labels:
        num_files = len(os.listdir(train_path + label))
        print(f"Number of {label[:-1]}s: {str(num_files)}")

# Sets up directories for train, test data 
def init_dirs():
    remove_dir(train_path)
    remove_dir(test_path)
    make_dir(train_path)
    make_dir(test_path)
    for label in labels:
        make_dir(train_path+label)
        # make_dir(test_path+label) # Not needed if sliding window

# Preprocesses raw data into train data
def setup_train_data(raw_path,label):
    files = os.listdir(raw_path +label)
    # Read each file 
    for i in range(len(files)):
        file_name = files[i]
        # Input raw file
        print("Input:" , file_name)
        file_path = raw_path +label + file_name
        # Output file path 
        out_file = train_path+label+label[:-1] + "_" + str(i)+".txt"  
        try:
            df = pd.read_csv(file_path, header=None, delimiter='\n')
            # only need <timesteps> rows
            df = df.head(timesteps)
            if not df.empty and not df.shape[0] < timesteps:
                print("Output:" , out_file)
                df.to_csv(out_file,float_format='%.4f')
        except Exception as e:
            print(e)

# Clears directory
def remove_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)

# Makes directory 
def make_dir(path):
    if not os.path.isdir(path):
        print("Making dir.... " + path)
        os.mkdir(path)
        

if __name__ == "__main__":
    main()