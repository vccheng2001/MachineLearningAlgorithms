import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from scipy.stats import norm
from pyts.approximation import SymbolicAggregateApproximation
import pandas as pd 

hr_file = "MIMIC_7633/7633_heartrate.csv"
rr_file = "MIMIC_7633/7633_resprate.csv"  
sbp_file = "MIMIC_7633/7633_systolicbp.csv"  

# Read input file 
def read_file(file_name):
    # Read from csv
    df = pd.read_csv(file_name,delimiter=',')
    X = df.values  # numpy array
    return X, len(X)

# Normalize 
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
 
# Labels time series data with <n_bins> characters 
def main():
    # File Name
    file_name = hr_file
    X, lengthX = read_file(file_name)
    n_samples, n_timestamps = 1, lengthX
    X = X.reshape((n_samples,n_timestamps))
    X = NormalizeData(X)
    
    # SAX transformation
    n_bins = 12
    sax = SymbolicAggregateApproximation(n_bins=n_bins, strategy='uniform')
    X_sax = sax.fit_transform(X)

    # Compute gaussian bins
    bins = norm.ppf(np.linspace(0,1, n_bins + 1)[1:-1])

    # Show the results for the first time series
    bottom_bool = np.r_[True, X_sax[0, 1:] > X_sax[0, :-1]]
    
    #Plot 
    chars = plot_SAX(file_name, X, X_sax, bottom_bool,n_timestamps, n_bins, bins)
  
    # Find most frequent sequences of length 4
    find_freq(chars, 4)

# Finds most frequent sequences 
def find_freq(chars, n):
    countDict = {}
    sublists = []
    # Appends all possible sublist sequences 
    for i in range(len(chars)):
        sublists.append(tuple(chars[i:i+n]))
    # Each dict key is a sequence
    for s in sublists:
        countDict[s] = 0
    # Iterate through char sequence, add sequences to dict 
    for i in range(len(chars)-n+1):
        key = tuple(chars[i:i+n])
        countDict[key] = countDict[key] + 1
    # Print sequences 
    for k, v in sorted(countDict.items(), key=lambda item: item[1]):
        print(k, v)

# PLot SAX
def plot_SAX(file_name, X, X_sax, bottom_bool,n_timestamps, n_bins, bins):
    # Stores all characters in time series order 
    chars = []
    plt.figure(figsize=(18, 7))
    plt.plot(X[0], 'o--', label='Original')
    for x, y, s, bottom in zip(range(n_timestamps), X[0], X_sax[0], bottom_bool):
        chars.append(s)
        va = 'bottom' if bottom else 'top'
        plt.text(x, y, s, ha='center', va=va, fontsize=14, color='#ff7f0e')
    plt.hlines(bins, 0, n_timestamps, color='g', linestyles='--', linewidth=0.5)
    sax_legend = mlines.Line2D([], [], color='#ff7f0e', marker='*',
                            label='SAX - {0} bins'.format(n_bins))
    first_legend = plt.legend(handles=[sax_legend], fontsize=8, loc=(0.76, 0.86))
    ax = plt.gca().add_artist(first_legend)
    plt.legend(loc=(0.81, 0.93), fontsize=8)
    plt.xlabel('Time', fontsize=14)
    plt.title('Symbolic Aggregate approx. for ' + file_name, fontsize=16)
    plt.ylim((0,1))
    plt.show()
    return chars 

if __name__ == "__main__":
    main()