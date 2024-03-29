import sys
import numpy as np

# Hidden Markov Model
# Learns A (Transition matrix), B (Emission matrix), C (Prior probabilities )

def main():
    (program, train_input, index_to_word, index_to_tag, hmmprior, hmmemit, hmmtrans) = sys.argv
    
    # States/Tags
    states = np.loadtxt(index_to_tag, dtype='str')
    # Words 
    words = np.loadtxt(index_to_word, dtype='str') 
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
    
    # Build prior vector pi
    build_hmmprior(states, states_dict, len_states,  train_input,  hmmprior)
    # Build transition matrix A 
    build_hmmtrans(states, states_dict, len_states, train_input, hmmtrans)
    # Build emission matrix B
    build_hmmemit(states, states_dict, len_states, words, words_dict, len_words, train_input, hmmemit)


# Build prior vector pi
def build_hmmprior(states, states_dict, len_states, input_file,output_file):
    prior = np.zeros(len_states)
    f = open(input_file, "r")
    # Init states 
    for row in f:
        first = row.split()[0]
        first_tag = first.split('_')[1]
        index = states_dict[first_tag]
        prior[index] += 1
    # add pseudocount 
    prior_with_pseudo = (prior + 1)
    # sum over all states 
    total_sum = np.sum(prior_with_pseudo)
    # prob: N(y1=sj)+1/sum(N(y1=sp) + 1))
    prior = np.divide(prior_with_pseudo, total_sum)
    # Write to hmmprior output file 
    np.savetxt(output_file, prior, delimiter=" ")
    f.close()

# Build A (Transition) matrix (len_states x len_words)
def build_hmmtrans(states, states_dict, len_states, input_file, output_file):
    trans = np.zeros((len_states, len_states))
    f = open(input_file, "r")
    # Number of times sj is associated with the word k
    for row in f: 
        sequence = row.split()
        # Get each curr_state, next_state pair in sequence 
        for i in range(len(sequence) - 1):
            curr_state = sequence[i].split('_')[1]
            next_state = sequence[i+1].split('_')[1]
            # Map to state indices 
            j = states_dict[curr_state]  # curr state 
            k = states_dict[next_state] # next state
            trans[j][k] += 1 # state k after state j 
    # add pseudocount 
    trans_with_pseudo = (trans + 1)
    print(trans_with_pseudo)
    # sum over states
    sum_over_states = trans_with_pseudo.sum(axis=1, keepdims=True)
    trans = np.divide(trans_with_pseudo, sum_over_states)
    # Write to output trans file 
    np.savetxt(output_file, trans, delimiter=" ")
    f.close()   


# Build B (Emission) matrix (len_states x len_words)
def build_hmmemit(states, states_dict, len_states, words, words_dict, len_words, input_file, output_file):
    emit = np.zeros((len_states, len_words))
    f = open(input_file, "r")
    # Number of times sj is associated with the word k
    for row in f: 
        sequence = row.split()
        for seq in sequence:
            (word, state) = seq.split('_')
            j = states_dict[state]
            k = words_dict[word]
            emit[j][k] += 1 # see word k at state j 
    # add pseudocount 
    emit_with_pseudo = (emit + 1)
    # sum over states
    sum_over_states = emit_with_pseudo.sum(axis=1, keepdims=True)
    # prob: N(y1=sj)+1/sum(N(y1=sp) + 1))
    emit = np.divide(emit_with_pseudo, sum_over_states)
    # Write to output emit file 
    np.savetxt(output_file, emit, delimiter=" ")
    f.close()            

        
   

if __name__ == "__main__":
    main()