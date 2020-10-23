import csv
import sys

# Vivian Cheng, 10-301 HW1 Programming: Decision Stump

## Opens up csv file
def open_file(input_file, split_index):
    input = open(input_file)
    D = csv.reader(input, delimiter = "\t")
    headings = next(D)
    feature_to_split_by = headings[split_index]
    return D

## Partitions dataset D into D0, D1 by splitting on m
def split_dataset(D, split_index, feature_labels):
    val_1 = feature_labels[1]
    D0, D1 = [], []
    ## Partition into D0, D1 by splitting on m
    for row in D:
        if row[split_index] in val_1: ## binary 1
            D1.append(row)
        else: ## binary 0
            D0.append(row)
    return (D0, D1)

## Retrieves majority vote for partitioned D0, D1
def get_majority_vote(dataset, split_index):
    votes = {}
    for x_vec in dataset:
        label = x_vec[-1]
        votes[label] = votes.get(label, 0) + 1
    votes = sorted(votes, key = votes.get, reverse = True)
    return votes[0]

## Writes predictions to output file
def make_predictions(out_file, dataset, split_index, majority_votes, feature_labels):
    (val_0, val_1) = feature_labels[0], feature_labels[1]
    (v0, v1) = majority_votes
    errors, dataset_size = 0.0, 0.0
    with open(out_file, 'w') as out:
        for row in dataset:
            dataset_size += 1
            if row[split_index] in val_1:
                out.write(v1 + "\n") ## If predicted != actual, error
                errors = (errors + 1) if (v1 != row[-1]) else errors
            elif row[split_index] in val_0:
                out.write(v0 + "\n") ## If predicted != actual, error
                errors = (errors + 1) if (v0 != row[-1]) else errors
    out.close()
    ## Error rate: num errors / total rows
    error_rate = errors / dataset_size
    return error_rate

## Output errors to metrics_out file
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
    (program, trn_in, test_in, split_index, trn_out, test_out, metrics_out) = sys.argv
    split_index = int(split_index)
    ## Open file, capture dataset D
    train_dataset = open_file(trn_in, split_index)
    ## Feature label: binary 1 value, binary 0 value
    feature_labels = {0: ['n', 'notA'], 1: ['y', 'A']}
    ## Partition dataset D into D0, D1 based on feature m
    (D0, D1) = split_dataset(train_dataset, split_index, feature_labels)
    ## Majority vote for partitioned datasets D0, D1
    v0 = get_majority_vote(D0, split_index)
    v1 = get_majority_vote(D1, split_index)
    majority_votes = (v0, v1)
    ## Make predictions for training dataset
    train_dataset = open_file(trn_in, split_index)
    train_error = make_predictions(trn_out, train_dataset, split_index, majority_votes, feature_labels)
    ## Make predictions for test dataset
    test_dataset = open_file(test_in, split_index)
    test_error = make_predictions(test_out, test_dataset, split_index, majority_votes, feature_labels)
    ## Output metrics
    output_error(metrics_out, train_error, test_error)

if __name__ == "__main__":
    main()