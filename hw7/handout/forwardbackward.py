import sys
import numpy as np
import matplotlib.pyplot as plt
# Performs forward-backward algorithm 
def main():
    (program, input_file, valid_file, index_to_word, index_to_tag, hmmprior, hmmemit, hmmtrans, predicted_file, metrics_file) = sys.argv
    # Tags  (states Y)
    states = np.loadtxt(index_to_tag, dtype='str')
    # Words (observations X)
    words = np.loadtxt(index_to_word, dtype='str') 

    # # Get prior pi, trans A, emit B 
    # prior = np.loadtxt(hmmprior, dtype=np.float64) 
    # A = np.loadtxt(hmmtrans, dtype=np.float64) 
    # B = np.loadtxt(hmmemit, dtype=np.float64)  

    # N, M
    len_states, len_words = np.size(states), np.size(words)
    states_dict, word_dict= {}, {}
    # put into states dict, map state to index 
    for i in range(len_states):
        state = states[i]
        states_dict[state] = i 
    # put into words dict, map word to index 
    for i in range(len_words):
        word = words[i]
        word_dict[word] = i

    # Input file (train/validation)
    input = open(input_file, "r")   
    valid = open(valid_file, 'r')
    # Prediction file    
    output= open(predicted_file, 'w')
    # Metrics file 
    metrics = open(metrics_file, 'w')

    seq_array =[10, 100, 1000, 10000]

    avgLL_array = []
    valid_avgLL_array = []
    for numseq in seq_array:
        hmmprior = hmmprior.split(".")[0]
        hmmtrans = hmmtrans.split(".")[0]
        hmmemit = hmmemit.split(".")[0]
        # Load pi/A/B
        prior = np.loadtxt(hmmprior+str(numseq)+'.txt', dtype=np.float64) 
        A = np.loadtxt(hmmtrans+str(numseq)+'.txt', dtype=np.float64) 
        B = np.loadtxt(hmmemit+str(numseq)+'.txt', dtype=np.float64)  

        avgLL = makePredictions(numseq, prior, A, B, len_states, len_words, states, states_dict, word_dict, input_file)
        valid_avgLL = makePredictions(numseq, prior, A, B, len_states, len_words, states, states_dict, word_dict, valid_file)

        avgLL_array.append(avgLL)
        valid_avgLL_array.append(valid_avgLL)
    # Metrics predictions 
    # metrics.write("Average Log-Likelihood: %s\n" % str(averageLogLikelihood))
    # metrics.write("Accuracy: %s\n" % str(numCorrect/sumTs) ) 

    print(seq_array) 
    print(avgLL_array)
    print(valid_avgLL_array)
    plt.plot(seq_array, avgLL_array, color="blue", label="Train")
    plt.plot(seq_array, valid_avgLL_array, color="red", label="Validation")
    plt.ylabel("Average Log Likelihood")
    plt.xlabel("# Sequences")
    plt.title("HMM")
    plt.show()

def makePredictions(numseq, prior, A, B, len_states, len_words, states,states_dict, word_dict, input_file):
    input = open(input_file, 'r')
    # Keep track of statistics 
    sumTs = 0
    sumLogLikelihood = 0
    numRows = 0
    numCorrect = 0
    # For each sequence
    for i in range(numseq):
        row = input.readline()
        numRows += 1
        sequence = row.split()
        maxT = len(sequence) # Number of words/tags in the sequence 
        sumTs += maxT  
        # Build alpha
        alphaMatrix = build_alpha(sequence, word_dict, maxT, states, len_states, prior, B, A)
        # print(alphaMatrix)
        # Build Beta
        BetaMatrix = build_Beta(sequence,word_dict,maxT, states, len_states, prior, B, A)
        # print(BetaMatrix)
        # For each 'timestep' t
        for t in range(maxT):
            # Get prob 
            alphaBeta = np.add(alphaMatrix[t], BetaMatrix[t])
            # State that yields highest prob (argmax)
            max_index = np.argmax(alphaBeta)
            # Predict new tag for the word 
            (orig_word, orig_tag) = sequence[t].split('_')
            new_tag = states[max_index].strip()
            # output.write((orig_word + "_" + new_tag))
            # If correct 
            if orig_tag == new_tag:
                numCorrect += 1
            # Write prediction 
        #     if t != maxT - 1:
        #         output.write(" ")
        # output.write('\n')
        # Add log likelihood contribution for sequence 
        sumLogLikelihood += logsumexp(alphaMatrix[maxT-1])
    print("NUMROWS", numRows)
    # Average log likelihood over all sequences 
    averageLogLikelihood = sumLogLikelihood / numRows
    print("ACCURACY", str(numCorrect/sumTs))
    input.close()
    return averageLogLikelihood


# Build alpha
def build_alpha(sequence, word_dict, maxT,states, len_states, prior, B,A):
    alphaMatrix = np.zeros((1,len_states))
    # starting t
    t = 0
    # Recursive helper function
    alphaMatrix = build_alpha_helper(sequence, word_dict, t, maxT, alphaMatrix, states, len_states,  prior, B,A)
    return alphaMatrix

# Alpha helper function
def build_alpha_helper(sequence,word_dict, t, maxT, alphaMatrix, states, len_states,  prior, B,A):
    alphaVec = np.zeros(len_states)
    # Exit case 
    if (t == maxT): return alphaMatrix
    # Get index of current word in sequence 
    word = sequence[t].split('_')[0]
    index = word_dict[word]
    # Base Case, start from t = 0
    if (t == 0): 
        # p(starting state is j) * p(see observation 0 at state j)
        alphaVec= (np.log(B[:,index]) + np.log(prior))
        # Update all states for timestep 0
        alphaMatrix[0] = alphaVec
        return build_alpha_helper(sequence,word_dict, t+1, maxT, alphaMatrix, states, len_states,  prior, B,A) 
    # Recurse forwards, t > 0
    m = np.max(alphaMatrix[t-1])
    alphaVec = np.log(B[:,index]) + m + np.log(np.dot(A.T, np.exp(alphaMatrix[t-1] - m)))
    #np.multiply(B[:,index], np.dot(A.T, alphaMatrix[t-1]))
    # Append vector to matrix 
    alphaMatrix = np.vstack((alphaMatrix, alphaVec))
    return build_alpha_helper(sequence,word_dict, t+1, maxT,alphaMatrix, states,len_states, prior, B,A) 


def logsumexp(v):
    m = np.max(v)
    return m + np.log(np.sum(np.exp(v-m)))


# Build Beta
def build_Beta(sequence,word_dict,maxT, states, len_states, prior, B,A):
    # Init BetaMatrix
    BetaMatrix = np.zeros((1,len_states))
    # Start with timestep T - 1
    t = maxT - 1
    # Recursive Beta helper
    BetaMatrix = build_Beta_helper(sequence,word_dict,t, maxT, BetaMatrix, states, len_states,  prior, B,A)
    return BetaMatrix

# Build Beta
def build_Beta_helper(sequence,word_dict,t, maxT, BetaMatrix, states, len_states,  prior, B,A):
    BetaVec = np.zeros(len_states)
    # Exit case
    if (t < 0): return BetaMatrix
    # Base case, start from maxT - 1 (end)
    if (t == maxT - 1):
        BetaVec = np.log(np.ones(len_states))
        BetaMatrix[0] = BetaVec 
        return build_Beta_helper(sequence,word_dict,t-1, maxT, BetaMatrix, states, len_states,  prior, B,A) 
    # Get index of current word in sequence 
    word = sequence[t+1].split('_')[0]
    index = word_dict[word]
    # Recurse backwards 
    m = np.max(BetaMatrix[0])
    BetaVec = m+np.log(np.dot(A, np.add(B[:,index],np.exp(BetaMatrix[0] - m))))

    #BetaVec = np.dot(A, np.multiply(B[:,index], BetaMatrix[0]))
    # Append vector to matrix 
    BetaMatrix = np.vstack((BetaVec, BetaMatrix)) 
    return build_Beta_helper(sequence,word_dict,t-1,  maxT, BetaMatrix, states,len_states, prior, B,A)
   
if __name__ == "__main__":
    main()