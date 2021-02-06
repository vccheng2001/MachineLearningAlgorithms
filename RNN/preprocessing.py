
import pandas as pd 
import os
import numpy as np
import csv
import shutil
apnea_type = "osa"


raw_path = "raw_" + apnea_type + '/'
train_path = "train_" + apnea_type + '/'
test_path = "test_" + apnea_type + '/'

timesteps = 160
# Preprocesses apnea files 
def main():
    remove_dir(train_path)
    remove_dir(test_path)
    make_dir(train_path)
    make_dir(test_path)
    for group in ['positive/', 'negative/']:
        make_dir(train_path+group)
        make_dir(test_path+group)
    for group in ["positive/", "negative/"]:
        read_files(raw_path, group)
    # for group in ["positive/", "negative/"]: 
    #     num_files = len(os.listdir(train_path + group))
    #     print("Number of " + group[:-1] + "s: " + str(num_files))


def read_files(raw_path, group):
    # List all files in group 
    files = os.listdir(raw_path + group)
    # Read each file 
    for i in range(len(files)):
        file_name = files[i]
        print("Input:" , file_name)
        # Load format
        file_path = raw_path + group + file_name
        out_file = train_path + group + group[:-1] + "_" + str(i)+".txt"  
        try:
            data = np.genfromtxt(file_path, delimiter="\n",dtype=None)
            np.savetxt(out_file, data, delimiter='\n', fmt='%.4f ')
            print("Output: " , out_file)
            # Delete nans, resize 
            df = pd.read_csv(out_file, delimiter='\n',names=["value"])
            # Method 1: Rescale between range (predicts many false positives)
            # a, b = 0, 1000
            # x, y = df["value"].min(), df["value"].max()
            # df["value"] = (df["value"] - x) / (y - x) * (b - a) + a

            # Method 2: normalize (not accurate )
            # df=(df-df.mean())/df.std()

            # Method 3: Rolling window (predicts all 0s) 
            # df.rolling(window=2).mean()
            df = df.head(timesteps)
            if df.empty or df.shape[0] < timesteps:
                os.remove(out_file)
            else:
                df.to_csv(out_file,index=False,header=None)
        except Exception as e:
            os.remove(out_file)
            print(e)
        
def remove_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)

def make_dir(path):
    if not os.path.isdir(path):
        print("Making dir.... " + path)
        os.mkdir(path)
        

if __name__ == "__main__":
    main()