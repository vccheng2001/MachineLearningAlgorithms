
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
        df = pd.read_csv(file_path, sep=",", usecols=[1])
        df = df.rolling(10).mean() 
        df = df.dropna()
        df = df.iloc[::10, :]
        df = df.head(400-1)
        if df.shape[0] < 400-1: continue
        else:
            df.to_csv(train_path + group + group[:-1] + "_" + str(i)+".txt", index=False)
     # for file_name in files:
    #     print(raw_path + group + file_name)
    #     file_path = raw_path + group + file_name 
    #     df = pd.read_csv(file_path, sep=",", usecols=[1])
    #     # First 3000 rows
    #     df = df.head(3000-1)
    #     # df = windows(df, 128, 64)
    #     df.to_csv(train_path + group + file_name, index=False)

# d: dataframe
# w: window size (# timesteps) = 128
# t: overlapping factor (50%)  = 64
def windows(d, w, t):  
    r = np.arange(len(d))   
    s = r[::t]   
    z = list(zip(s, s + w))   
    f = '{0[0]}:{0[1]}'.format
    g = lambda t: d.iloc[t[0]:t[1]]   
    return pd.concat(map(g, z), keys=map(f, z)) 


if __name__ == "__main__":
    main()