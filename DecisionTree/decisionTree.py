import sys
import csv
import math

''' 
This program builds a decision tree with maximum depth max_depth given input 
tsv files with a dataset D. It determines which attributes to split by 
by calculating mutual information gain at each split.
'''

## Node class, each node stores data D, depth, and left/right children nodes
class Node:
    def __init__(self, D, depth):
        self.D = D
        self.depth = depth
        self.right = None
        self.left = None
        self.name = []
        self.prediction = None

## Retrieve binary labels/output classes
def get_labels(D):
    labels = []
    D_size = len(D)
    for row in D:
        if row[-1] not in labels:
            labels.append(row[-1])
    return sorted(labels, reverse=False)

## Retrieve binary feature values
def get_feature_vals(D):
    feature_vals = []
    D_size = len(D)
    for row in D:
        for val in row[:-1]:
            if val not in feature_vals:
                feature_vals.append(val)
    return sorted(feature_vals)


## Helper function: Return count of an output label/class in dataset D
def count(D, label):
    count = 0
    for row in D:
        if row[-1] == label:
            count += 1
    return count

## Calculate entropy for a given dataset D
def entropy(D, labels):
    D_size = len(D)
    entropy = 0
    for l in labels:
        label_count = count(D, l)
        if label_count > 0:
            p = label_count / D_size # Probability of each label
            entropy += (-p * math.log2(p)) # sum (-plog2(p))
    return entropy

## Returns mutual info gain for a node
def mutual_info(node, f_index, f_vals, classes):
    ## split into left/right child datasets
    (D0, D1) = split_dataset(node, f_index, f_vals, classes)
    ## get sizes of parent, left child, right child datasets
    D_size, D0_size, D1_size = len(node.D), len(D0), len(D1)
    ## get probabilities
    p_D0, p_D1 = (D0_size / D_size), (D1_size / D_size) # prob of D0, D1
    ## calculate info gain I = H(Y) - H(Y|X)
    mutual_info_gain = entropy(node.D, classes)- ((p_D0 * entropy(D0, classes)) + (p_D1 * entropy(D1, classes)))
    return mutual_info_gain

## Select best attribute to split on
def pick_best_feature(node, F, f_vals, classes):
    best_index, best_feature, max_mi_gain = None, None, 0
    ## Calculate mutual info that results from splitting on each feature
    for f_index, f_name in F.items():
        mi_gain = mutual_info(node, f_index, f_vals, classes)
        ## Store attribute that produces highest info gain
        if mi_gain >= max_mi_gain:
            max_mi_gain = mi_gain
            best_feature = f_name
            best_index = f_index
    ## Only split by attribute if mutual info gain > 0
    return (best_index, best_feature) if max_mi_gain > 0 else (None, None)

## Partitions node's dataset into D0, D1 by splitting on feature f
def split_dataset(node, f_index, f_vals, classes):
    (v0, v1) = f_vals ## values that feature can take on
    D0, D1 = [], []
    ## Partition into D0, D1
    for row in node.D:
        if row[f_index] == v1:## binary 1
            D1.append(row)
        else: ## binary 0
            D0.append(row)
    return (D0, D1)

## Returns majority vote for a node (most probable class)
def majority_vote(node, predictions):
    votes = {}
    if not node.D:
        return None
    for row in node.D:
        label = row[-1]
        votes[label] = votes.get(label, 0) + 1
    ## Ranking of output classes
    ranking = sorted(votes, key = votes.get, reverse = True)
    ## Sort in lexographically reversed order if tie 
    if len(ranking) == 2:
        if votes[ranking[0]] == votes[ranking[1]]:
            ranking = sorted(ranking, reverse=True)
    ## most probable class for leaf node
    prediction = ranking[0]
    ## Number of mispredictions
    node.prediction = prediction
    predictions.append([node.name, node.prediction])
    return prediction

## Build tree from node
def build_tree(node, F, f_vals, classes, max_depth, predictions):
    [label0, label1] = classes ## output classes
    ## at depth 0, print count of output labels for full dataset
    if node.depth == 0:
        print("[%d %s/%d %s]" % (count(node.D, label0), label0 , count(node.D, label1), label1))
    ## base case: if node is perfectly classified or max depth reached
    if entropy(node.D, classes) == 0 or node.depth == max_depth:
        return majority_vote(node, predictions)
    ## choose feature giving lowest error rate (only split if info gain > 0)
    (f_index, f) = pick_best_feature(node, F, f_vals, classes)
    if not f_index and not f:
        return majority_vote(node, predictions)

    ## partition D on m, D0 has all rows where f == 0, D1 has all rows where f == 1
    (D0, D1) = split_dataset(node, f_index, f_vals, classes)
    ## create left, right children node
    node.left = Node(D0, node.depth + 1)
    node.left.name = node.name + [(f_index, f_vals[0])]
    node.right=  Node(D1,  node.depth + 1)
    node.right.name = node.name + [(f_index, f_vals[1])]

    ## build tree from right node
    print_dtree(node, f, f_vals[1], D1, classes)
    build_tree(node.right, F, f_vals, classes, max_depth, predictions)

    ## build tree from left node
    print_dtree(node, f, f_vals[0], D0, classes)
    build_tree(node.left, F, f_vals, classes, max_depth, predictions)

## Print out decision tree at each depth
def print_dtree(node, f, f_val, D, labels):
    print("| " * (node.depth+1) + "%s = %s : " % (f, f_val), end="")
    print("[%d %s/%d %s]" % (count(D, labels[0]), labels[0] , count(D, labels[1]), labels[1]))

## Given data D, build a decision tree
def decisionTree(D, F, f_vals, classes, max_depth, predictions):
    ## Root node with full dataset D
    root = Node(D, 0)
    return build_tree(root, F, f_vals, classes, max_depth, predictions)

# Output errors to metrics_out file
def output_error(metrics_out, train_error, test_error):
    with open(metrics_out, 'w') as metrics:
        train_string = "error(train): %f\n" % train_error
        test_string = "error(test): %f\n" % test_error
        metrics.write(train_string)
        metrics.write(test_string)
    metrics.close()


## Main function
def main():
    ## Parse args
    (program, trn_in, test_in, max_depth, trn_out, test_out, metrics_out) = sys.argv
    ## Open file, capture full dataset D
    inp = open(trn_in)
    D = list(csv.reader(inp, delimiter = "\t"))
    ## Store all features from csv header into a dictionary F
    features_list = D.pop(0)[:-1]
    F, f_index = {}, 0
    for f_index in range(len(features_list)):
        F[f_index] = features_list[f_index]
    # Get feature values, output class labels
    classes = get_labels(D)
    f_vals = get_feature_vals(D)
    ## Build a decision tree using input dataset D
    predictions = []
    decisionTree(D, F, f_vals, classes, int(max_depth), predictions)
    ## Make predictions for training dataset
    train_input = open(trn_in)
    train_error = make_prediction(train_input, trn_out, predictions)
    ## Make predictions for test dataset
    test_input = open(test_in)
    test_error = make_prediction(test_input, test_out, predictions)
    ## Output train and test error
    output_error(metrics_out, train_error, test_error)

## Makes prediction based on decision tree model 
def make_prediction(in_file, out_file, predictions):
    prediction = None
    errors, D_size = 0.0, 0.0
    D = csv.reader(in_file, delimiter = "\t")
    headings = next(D)
    ## Opens output file to write to 
    with open(out_file, 'w') as out:
        for row in D:
            D_size += 1
            actual = []
            ## Records actual values 
            for i in range(len(row) - 1):
                actual.append((i, row[i]))
            ## Makes prediction based on stored predictions from decision tree 
            for p in predictions:
                prediction = p[1] if contains_all(p[0], actual) else prediction
            out.write(prediction+ "\n")
            ## If mismatch actual/prediction, error count increments 
            errors = (errors + 1) if (prediction != row[-1]) else errors
    out.close()
    error_rate = errors / D_size
    return error_rate

## Helper function to check actual data against prediction by 
## decision tree for a given row 
def contains_all(pred, actual):
    for p in pred:
        if p not in actual:
            return False 
    return True

## Output errors to metrics_out file
def output_error(metrics_out, train_error, test_error):
    with open(metrics_out, 'w') as metrics:
        train_string = "error(train): %f\n" % train_error
        test_string = "error(test): %f\n" % test_error
        metrics.write(train_string)
        metrics.write(test_string)
    metrics.close()

if __name__ == "__main__":
    main()
