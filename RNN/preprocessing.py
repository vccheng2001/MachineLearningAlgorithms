
import pandas as pd 
import os
import numpy as np

raw_path = "raw/"
train_path = "train/"
hello_path = "hello/"
# Preprocesses apnea files 
def main():
    raw_path = "raw/"
    for group in ["positive/", "negative/"]:
        read_files(raw_path, group)
    
def read_files(raw_path, group):
    # List all files in group 
    files = os.listdir(raw_path + group)
    # Read each file 
    for i in range(len(files)):
        file_name = files[i] 
        file_path = raw_path + group + file_name
        df = pd.read_csv(file_path, sep=",", names=["time", "value"])

        # Processing
        # df['value'] = df['value'].rolling(30).mean() #rolling average every half second 
        df = df.dropna()
        df = df.iloc[::20, :]
        df = df.head(300)
        if df.shape[0] < 300: 
            continue
        else:
            df.to_csv(train_path + group + group[:-1] + "_" + str(i)+".txt", index=False,header=None)


if __name__ == "__main__":
    main()