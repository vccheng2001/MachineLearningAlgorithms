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
    for row in f:
        sequence = row.split()
        # print(sequence)
        maxT = len(sequence) 
        print(sequence)
        for t_state in range(maxT):
            # Build alpha 
            alphaMatrix = build_alpha(t_state, states, len_states, prior, B, A)
            BetaMatrix = build_Beta(t_state, maxT, states, len_states, prior, B, A)
            alphaBeta = np.multiply(alphaMatrix[t_state - 1], BetaMatrix[t_state - 1])
            max_index = np.argmax(alphaBeta)
            #print(sequence[t_state].split('_')[0] + "_" + states[max_index], end=" ")
        exit(0)
        # print('\n')


# Build alpha
def build_alpha(t_state, states, len_states, prior, B,A):
    # print('------------------alpha--------------------')
    # print('t_state', t_state, '\n')
    print('prior', prior,'\n')
    print(prior.shape)
    print('B', B, '\n')
    print('A', A,'\n')
    # init 
    alphaMatrix = np.zeros((1,len_states))
    # starting t
    t = 0
    alphaMatrix = build_alpha_helper(t, t_state, alphaMatrix, states, len_states,  prior, B,A)
    # print("alphaaaa for ", t_state)
    # print(alphaMatrix)
    return alphaMatrix

# Build alpha helper function
def build_alpha_helper(t, t_state, alphaMatrix, states, len_states,  prior, B,A):
    alphaVec = np.zeros(len_states)
    # Exit case 
    if (t > t_state): return alphaMatrix

    # Base Case, start from t = 0
    if (t == 0): # corresponds to t == 1 (0)
        # p(starting state is j) * p(see observation 1 at state j)
        alphaVec= np.multiply(B[:,t], prior)
        # Update all states for timestep 1 
        alphaMatrix[0] = alphaVec
        return build_alpha_helper(t+1, t_state, alphaMatrix, states, len_states,  prior, B,A) 
    
    # Recurse t > 1 to T... 2 to 3 (1 to 2)
    # for j in range(len_states): 
    #     aAsum = 0
    #     for k in range(len_states):
    #         aAsum += np.dot(alphaMatrix[t-1][k], A[k][j])
    #     alphaVec[j] = np.multiply(aAsum, B[j][t])
    # append to old matrix
   
    # print(t)
    # print(B[:,t]) 
    # print(A)
    # print(alphaMatrix[t-1])
    alphaVec = np.multiply(B[:,t], np.dot(A.T, alphaMatrix[t-1]))
    # print('----')
    # print(alphaVec)
    # print(alphaVec)
    alphaMatrix = np.vstack((alphaMatrix, alphaVec))
    # print(alphaMatrix)
    #rint(alphaMatrix)
    return build_alpha_helper(t+1, t_state, alphaMatrix, states,len_states, prior, B,A) 


def build_Beta(t_state, maxT, states, len_states, prior, B,A):
    # init 
    BetaMatrix = np.zeros((maxT,len_states))
    # start with timestep T
    t = maxT - 1
    BetaMatrix = build_Beta_helper(t, t_state, maxT, BetaMatrix, states, len_states,  prior, B,A)
    return BetaMatrix

# Build Beta
def build_Beta_helper(t, t_state, maxT, BetaMatrix, states, len_states,  prior, B,A):
    BetaVec = np.zeros(len_states)
    # Exit
    if (t < t_state): return BetaMatrix
    # Base case, start from end (t=len_states)
    if (t == maxT - 1):
        BetaVec = np.ones(len_states)
        BetaMatrix[t] = BetaVec 
        return build_Beta_helper(t-1, t_state, maxT, BetaMatrix, states, len_states,  prior, B,A) 
    # Recurse 
    for j in range(len_states):
        bBsum = 0
        for k in range(len_states):
            bBsum += np.multiply(BetaMatrix[t+1][k], B[k][t+1])
        BetaVec[j] = np.dot(bBsum, A[j][k])
    BetaMatrix[t] = BetaVec
    # BetaMatrix = np.vstack((np.zeros(len_states), BetaMatrix))
    return build_Beta_helper(t-1, t_state, maxT, BetaMatrix, states,len_states, prior, B,A) 


   

if __name__ == "__main__":
    main()