
import pandas as pd 
import os
import numpy as np
import csv

raw_path = "raw/"
train_path = "train/"

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
        print("Input:" , file_name)


        # Load format
        file_path = raw_path + group + file_name
        data = np.genfromtxt(file_path, delimiter=" ",dtype=None)
        out_file = train_path + group + group[:-1] + "_" + str(i)+".txt" 
        np.savetxt(out_file, data, delimiter='\n', fmt='%d')

        print("Output: " , out_file)
        # Delete nans, resize 
        df = pd.read_csv(out_file, delimiter='\n')
        df = df.dropna()
        df = df.tail(100)
        if df.shape[0] < 100: 
            os.remove(out_file)
        else:
            df.to_csv(out_file,index=False,header=None)

        

if __name__ == "__main__":
    main()