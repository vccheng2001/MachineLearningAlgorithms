import sys
import pandas as pd
import numpy as np

def main():
    (program, train_input, index_to_word, index_to_tag, hmmprior, hmmemit, hmmtrans) = sys.argv
    states = np.loadtxt(index_to_tag, dtype='str')
    len_states = np.size(states)
    build_hmmprior(states, len_states,  train_input, index_to_word, index_to_tag, hmmprior)
    

def build_hmmprior(states, len_states, input_file, index_to_word, index_to_tag, output_file):
    prior = np.zeros(len_states)
    f = open(input_file, "r")
    # Init states 
    for row in f:
        first = row.split()[0]
        first_tag = first.split('_')[1]
        for i in range(len_states):
            if first_tag == states[i]:
                prior[i] += 1
    # add pseudocount 
    prior_with_pseudo = (prior + 1)
    # sum over all states 
    total_sum = np.sum(prior_with_pseudo)
    # prob: N(y1=sj)+1/sum(N(y1=sp) + 1))
    prior = np.divide(prior_with_pseudo, total_sum)
    np.savetxt(output_file, prior, delimiter=" ")


                

        
   

if __name__ == "__main__":
    main()