import sys
import numpy as np

# Performs forward-backward algorithm 
def main():
    (program, validation_input, index_to_word, index_to_tag, hmmprior, hmmemit, hmmtrans, predicted_file, metric_file) = sys.argv
    states = np.loadtxt(index_to_tag, dtype='str')
    words = np.loadtxt(index_to_word, dtype='str') 

    # Get prior pi, trans A, emit B 
    prior = np.loadtxt(hmmprior, dtype='float64') 
    trans = np.loadtxt(hmmtrans, dtype='float64')  
    emit = np.loadtxt(hmmemit, dtype='float64')  

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
    build_alpha(states, states_dict, len_states, words, words_dict, len_words, prior, emit,trans, validation_input, predicted_file)
    # Build Beta 
    # build_Beta(states, states_dict, len_states, words, words_dict, len_words, hmmprior, hmmemit, hmmtrans, validation_input, predicted_file)


def build_alpha(states, states_dict, len_states, words, words_dict, len_words, prior, B,A, validation_input, predicted_file):
    # init 
    alphaMatrix = np.zeros((1,len_states))
    # start with timestep 1
    alpha_matrix = build_alpha_helper(0, alphaMatrix, states, states_dict, len_states, words, words_dict, len_words, prior, B,A)
    print(alpha_matrix)
    
# Build alpha
def build_alpha_helper(t, alphaMatrix, states, states_dict, len_states, words, words_dict, len_words, prior, B,A):
    alphaVec = np.zeros(len_states)
    if (t == len_states): return alphaMatrix
    if (t == 0):
        # p(starting state is j) * p(see observation 1 at state j)
        alphaVec= np.multiply(B[:,t], prior)
        # Update all states for timestep 1 
        alphaMatrix[0] = alphaVec
        return build_alpha_helper(t+1, alphaMatrix, states, states_dict, len_states, words, words_dict, len_words, prior, B,A) 
    # Recurse 
    for j in range(len_states): 
        aAsum = 0
        for k in range(len_states):
            aAsum += np.dot(alphaMatrix[t-1][k], A[k][j])
        alphaVec[j] = np.multiply(aAsum, B[j][t])
    alphaMatrix = np.vstack((alphaMatrix, alphaVec))
    return build_alpha_helper(t+1, alphaMatrix, states, states_dict, len_states, words, words_dict, len_words, prior, B,A) 

        # # alphaMatrix[t][j] = np.multiply(B[j][t], np.dot(A[t-1][j], alphaMatrix[t-1]))
        # alphaMatrix[t] = np.multiply(B[:,t], np.dot(A[t][j], alphaMatrix[t-1]))

    # else:
    #     alpha[t][j] = B[j][xt] * np.multiply(A[k][j], alpha[t-1])
    #     return build_alpha_helper(t+1, alphaMatrix, states, states_dict, len_states, words, words_dict, len_words, prior, B,A)
    


    # emit = np.zeros((len_states, len_words))
    # f = open(input_file, "r")
    # # Number of times sj is associated with the word k
    # for row in f: 
    #     sequence = row.split()
    #     for seq in sequence:
    #         (word, state) = seq.split('_')
    #         j = states_dict[state]
    #         k = words_dict[word]
    #         emit[j][k] += 1 # see word k at state j 
    # # add pseudocount 
    # emit_with_pseudo = (emit + 1)
    # # sum over states
    # sum_over_states = emit_with_pseudo.sum(axis=1, keepdims=True)
    # # prob: N(y1=sj)+1/sum(N(y1=sp) + 1))
    # emit = np.divide(emit_with_pseudo, sum_over_states)
    # # Write to output emit file 
    # np.savetxt(output_file, emit, delimiter=" ")
    # f.close()            

        
   

if __name__ == "__main__":
    main()