import sys
import numpy as np

# Performs forward-backward algorithm 
def main():
    (program, validation_input, index_to_word, index_to_tag, hmmprior, hmmemit, hmmtrans, predicted_file, metric_file) = sys.argv
    states = np.loadtxt(index_to_tag, dtype='str')
    words = np.loadtxt(index_to_word, dtype='str') 

    # Get prior pi, trans A, emit B 
    prior = np.loadtxt(hmmprior, dtype='float64') 
    A = np.loadtxt(hmmtrans, dtype='float64')  
    B = np.loadtxt(hmmemit, dtype='float64')  

    len_states, len_words = np.size(states), np.size(words)
    states_dict, words_dict= {}, {}
    # put into states dict, map state to index 
    for i in range(len_states):
        state = states[i]
        states_dict[state] = i 
    # put into words dict, map word to index 
    for i in range(len_words):
        word = words[i]
        words_dict[word] = i

    # Build alpha 
    build_alpha(states, len_states, prior, B, A)
    # Build Beta 
    #build_Beta(states, len_states, prior, B, A)

def build_alpha(states, len_states, prior, B,A):
    # init 
    alphaMatrix = np.zeros((1,len_states))
    # start with timestep 1
    alpha_matrix = build_alpha_helper(0, alphaMatrix, states, len_states,  prior, B,A)
    print(alpha_matrix)

# Build alpha
def build_alpha_helper(t, alphaMatrix, states, len_states,  prior, B,A):
    alphaVec = np.zeros(len_states)
    if (t == len_states): return alphaMatrix
    if (t == 0):
        # p(starting state is j) * p(see observation 1 at state j)
        alphaVec= np.multiply(B[:,t], prior)
        # Update all states for timestep 1 
        alphaMatrix[0] = alphaVec
        return build_alpha_helper(t+1, alphaMatrix, states, len_states,  prior, B,A) 
    # Recurse 
    for j in range(len_states): 
        aAsum = 0
        for k in range(len_states):
            aAsum += np.dot(alphaMatrix[t-1][k], A[k][j])
        alphaVec[j] = np.multiply(aAsum, B[j][t])
    alphaMatrix = np.vstack((alphaMatrix, alphaVec))
    return build_alpha_helper(t+1, alphaMatrix, states,len_states, prior, B,A) 


        
   

if __name__ == "__main__":
    main()