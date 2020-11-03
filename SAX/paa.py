import numpy as np
import matplotlib.pyplot as plt
from pyts.approximation import PiecewiseAggregateApproximation as PAA
import pandas as pd 

def read_file():
    # Read from csv
    file_name = "rr_small.csv"
    rr = pd.read_csv(file_name, sep=',', usecols=['respiration'], squeeze=True)
    X = rr.values  # numpy array
    # Length 
    lengthX = len(X)
    # Reshape to 2D
    X = X.reshape((1, lengthX))
    # Parameters
    n_samples, n_timestamps = 1, lengthX
    return (X, n_samples, n_timestamps)
 

def main():
    X, n_samples, n_timestamps = read_file()
    # PAA transformation
    window_size = 8
    paa = PAA(window_size=window_size)
    X_paa = paa.transform(X)

    # Show the results for the first time series
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(n_timestamps), X[0], 'o-', label='Original')
    plt.plot(np.arange(window_size // 2,
                    n_timestamps + window_size // 2,
                    window_size), X_paa[0], 'o--', label='PAA')
    plt.vlines(np.arange(0, n_timestamps, window_size),
            X[0].min(), X[0].max(), color='g', linestyles='--', linewidth=0.5)
    plt.legend(loc='best', fontsize=14)
    plt.show()

if __name__ == "__main__":
    main()
