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
    #build_alpha(states, len_states, prior, B, A)
    # Build Beta 
    build_Beta(states, len_states, prior, B, A)

def build_alpha(states, len_states, prior, B,A):
    # init 
    alphaMatrix = np.zeros((1,len_states))
    # start with timestep 1
    t = 0
    alpha_matrix = build_alpha_helper(t, alphaMatrix, states, len_states,  prior, B,A)

# Build alpha
def build_alpha_helper(t, alphaMatrix, states, len_states,  prior, B,A):
    alphaVec = np.zeros(len_states)
    # Exit 
    if (t == len_states): return alphaMatrix
    # Base Case, start from t=0
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


def build_Beta(states, len_states, prior, B,A):
    # init 
    BetaMatrix = np.zeros((len_states,len_states))
    # start with timestep T
    t = len_states - 1
    BetaMatrix = build_Beta_helper(t, BetaMatrix, states, len_states,  prior, B,A)

# Build Beta
def build_Beta_helper(t, BetaMatrix, states, len_states,  prior, B,A):
    BetaVec = np.zeros(len_states)
    # Exit
    if (t < 0): 
        BetaMatrix = np.delete(BetaMatrix, 0, 0)
        return BetaMatrix
    # Base case, start from end (t=len_states)
    if (t == len_states - 1):
        BetaVec = np.ones(len_states)
        BetaMatrix[t] = BetaVec 
        return build_Beta_helper(t-1, BetaMatrix, states, len_states,  prior, B,A) 
    # Recurse 
    for j in range(len_states):
        bBsum = 0
        for k in range(len_states):
            bBsum += np.multiply(BetaMatrix[t+1][k], B[k][t+1])
        BetaVec[j] = np.dot(bBsum, A[j][k])
    
    BetaMatrix[0] = BetaVec
    BetaMatrix = np.vstack((np.zeros(len_states), BetaMatrix))
    return build_Beta_helper(t-1, BetaMatrix, states,len_states, prior, B,A) 


   

if __name__ == "__main__":
    main()