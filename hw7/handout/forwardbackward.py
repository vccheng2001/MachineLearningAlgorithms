import sys
import numpy as np

# Performs forward-backward algorithm 
def main():
    (program, input_file, index_to_word, index_to_tag, hmmprior, hmmemit, hmmtrans, predicted_file, metric_file) = sys.argv
    states = np.loadtxt(index_to_tag, dtype='str')
    words = np.loadtxt(index_to_word, dtype='str') 

    # Get prior pi, trans A, emit B 
    prior = np.loadtxt(hmmprior, dtype=np.float64) 
    A = np.loadtxt(hmmtrans, dtype=np.float64) 
    B = np.loadtxt(hmmemit, dtype=np.float64)  
    # print(prior)
    # print(A)
    # print(B)
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

    # Estimate yhat_t, assign tags based on state that max P(Yt=sj | x1:T)
    f = open(input_file, "r")
    out = open(predicted_file, 'w')
    for row in f:
        sequence = row.split()
        maxT = len(sequence) 
     
        alphaMatrix = build_alpha(maxT, states, len_states, prior, B, A)
        BetaMatrix = build_Beta(maxT, states, len_states, prior, B, A)
        for t in range(maxT):
            alphaBeta = np.multiply(alphaMatrix[t], BetaMatrix[t])
            max_index = np.argmax(alphaBeta)
            out.write((sequence[t].split('_')[0] + "_" + states[max_index].strip()))
        out.write(' \n')


# Build alpha
def build_alpha(maxT,states, len_states, prior, B,A):
    alphaMatrix = np.zeros((1,len_states))
    # starting t
    t = 0
    alphaMatrix = build_alpha_helper(t, maxT, alphaMatrix, states, len_states,  prior, B,A)
    return alphaMatrix

# Build alpha helper function
def build_alpha_helper(t,maxT, alphaMatrix, states, len_states,  prior, B,A):
    alphaVec = np.zeros(len_states)
    # Exit case 
   # if (t > t_state): return alphaMatrix
    if (t == maxT): return alphaMatrix
    # Base Case, start from t = 0
    if (t == 0): # corresponds to t == 1 (0)
        # p(starting state is j) * p(see observation 1 at state j)
        alphaVec= np.multiply(B[:,t], prior)
        # Update all states for timestep 1 
        alphaMatrix[0] = alphaVec
        return build_alpha_helper(t+1, maxT, alphaMatrix, states, len_states,  prior, B,A) 
    
    alphaVec = np.multiply(B[:,t], np.dot(A.T, alphaMatrix[t-1]))
    alphaMatrix = np.vstack((alphaMatrix, alphaVec))
    return build_alpha_helper(t+1, maxT,alphaMatrix, states,len_states, prior, B,A) 


def build_Beta( maxT, states, len_states, prior, B,A):
    # init 
    BetaMatrix = np.zeros((1,len_states))
    # start with timestep T
    t = maxT - 1
    BetaMatrix = build_Beta_helper(t, maxT, BetaMatrix, states, len_states,  prior, B,A)
    return BetaMatrix

# Build Beta
def build_Beta_helper(t, maxT, BetaMatrix, states, len_states,  prior, B,A):
    BetaVec = np.zeros(len_states)
    # Exit
    if (t < 0): return BetaMatrix
    # Base case, start from end (t=len_states)
    if (t == maxT - 1):
        BetaVec = np.ones(len_states)
        BetaMatrix[0] = BetaVec 
        return build_Beta_helper(t-1, maxT, BetaMatrix, states, len_states,  prior, B,A) 
    # Recurse 
    BetaVec = np.dot(A, (np.multiply(B[:,t+1], BetaMatrix[0])))
    BetaMatrix = np.vstack((BetaVec, BetaMatrix)) 
    return build_Beta_helper(t-1,  maxT, BetaMatrix, states,len_states, prior, B,A) 


   

if __name__ == "__main__":
    main()