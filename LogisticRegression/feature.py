import csv
import sys

# Vivian Cheng, 10-301 HW1 Programming: Decision Stump

def get_counts(i, review, vocab, feat_vec, label_dict, feature_flag): # fill counts
    label, all_words = review.split("\t")
    words = all_words.split(" ")
    label_dict[i] = label
    for word in words: # skip label
        if vocab.get(word) or vocab.get(word) == 0:
            dict_index = vocab[word] # index of word in dictionary 
            if feature_flag == '1':
                feat_vec[i][dict_index] = 1
            else:
                feat_vec[i][dict_index] = feat_vec[i].get(dict_index, 0) + 1



def fill_dict(dict):
    vocab = {}
    for d in dict:
        word, id = d.split()
        word, id = word.strip(), id.strip()
        vocab[word] = int(id)
    return vocab

def print_counts(feat_vec, label_dict, outfile):
    f = open(outfile, 'w')
    for i in range(len(feat_vec)):
        if feat_vec[i] == {}: continue 
        f.write("%s\t" % label_dict[i])
        k = 0
        for key, value in feat_vec[i].items():
            f.write('%s:%s' % (key, value))
            k += 1
            if k < len(feat_vec[i]):
                f.write("\t")
        f.write("\n")
    f.close()


def filter_model_2(feat_vec):
    for i in range(len(feat_vec)):
        for key, value in list(feat_vec[i].items()):
            if value >= 4:
                del feat_vec[i][key]
            else:
                feat_vec[i][key] = 1
    return feat_vec


## Opens up csv file
def main():   
    (program, trn_in, valid_in, test_in, dict_in, ftrn_out, fvalid_out, ftest_out, feature_flag) = sys.argv
    # Parse dict 
    d = open(dict_in)
    vocab = fill_dict(d)
    # Parse movie reviews 
    files = [(trn_in, ftrn_out), (valid_in, fvalid_out), (test_in, ftest_out)]
    for (in_file, out_file) in files:
        # Parse movie reviews 
        feat_vec = {}
        label_dict= {}
        reviews = open(in_file, 'r')
        if feature_flag == '1':
            i = 0
            for review in reviews:
                feat_vec[i] = {} # initialize 
                get_counts(i, review, vocab, feat_vec, label_dict, feature_flag)
                i += 1 
            print_counts(feat_vec, label_dict, out_file)
            #remove_tab(out_file)
        elif feature_flag == '2':
            i = 0
            for review in reviews:
                feat_vec[i] = {}
                get_counts(i, review, vocab, feat_vec, label_dict, feature_flag)
                i += 1
            feat_vec = filter_model_2(feat_vec)
            print_counts(feat_vec, label_dict, out_file)
            #remove_tab(out_file)


# def remove_tab(out_file):
#     with open(out_file) as file:
#         for line in file:
#             line = line.lstrip()

if __name__ == "__main__":
    main()