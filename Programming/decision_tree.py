import math # for logarithm calculations
from collections import Counter # for counting occurrences
import csv # for reading TSV files
import sys # for command line arguments
try:
    import matplotlib.pyplot as plt # for plotting
except ImportError:
    plt = None

class Node:
    """Class representing a node in the decision tree."""

    def __init__(self, attr=None, vote=None, neg_count=0, pos_count=0):
        self.left = None    # child for attribute value 0
        self.right = None   # child for attribute value 1
        self.attr = attr    # attribute index to split on (None for leaf)
        self.vote = vote    # prediction for leaf nodes
        self.neg_count = neg_count  # count of negative examples
        self.pos_count = pos_count  # count of positive examples

    def is_leaf(self):
        return self.attr is None

# Helper functions for decision tree construction
# Entropy calculation
""" Entropy H(X) = - sum(p(x) * log2(p(x))) for all x in X """
def entropy_from_counts(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        entropy -= p * math.log(p, 2)
    return entropy

# Entropy of labels
""" H(Y) --> entropy of the label distribution"""
def entropy(labels):
    return entropy_from_counts(Counter(labels).values())

# Conditional entropy H(Y|X)
""" H(Y|X) = sum(p(x) * H(Y|X=x)) for all x in X """
def conditional_entropy(attribute_values, labels):
    total = len(labels)
    if total == 0:
        return 0.0
    
    groups = {}
    for x, y in zip(attribute_values, labels): # zip to pair attribute values with labels
        groups.setdefault(x, []).append(y)

    cond_entropy = 0.0
    for ys in groups.values():
        px = len(ys) / total
        cond_entropy += px * entropy(ys)
    return cond_entropy

# Mutual information I(Y;X)
""" I(Y;X) = H(Y) - H(Y|X) """
def mutual_information(attribute_values, labels):
    return entropy(labels) - conditional_entropy(attribute_values, labels)

# Majority label
""" Return the majority label (0 or 1). In case of tie, return 1. """
def majority_label(labels):
    if not labels:
        return 1  # default to 1 if no labels
    
    counter = Counter(labels)
    if len(counter) == 1:
        return list(counter.keys())[0]
    
    # If tied, choose 1
    if counter[0] == counter[1]:
        return 1 # favor 1 in ties
    
    return counter.most_common(1)[0][0]

# Load data from TSV file
""" Load data from a TSV file. Returns headers and data as list of lists. """
def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        headers = next(reader)
        for row in reader:
            data.append([int(x) for x in row])
    return headers, data

# Split data based on attribute value
""" Split data into two subsets based on attribute index and value. """
def split_data(data, attr_idx, value):
    return [row for row in data if row[attr_idx] == value]

# Build decision tree
""" Recursively build the decision tree. """
def build_tree(data, attr_names, max_depth=None, info_threshold=0.0, min_split_size=0, current_depth=0):
    if not data:
        return Node(vote=1, neg_count=0, pos_count=0)
    
    # Extract labels (last column)
    labels = [row[-1] for row in data]
    neg_count = labels.count(0)
    pos_count = labels.count(1)
    
    # Base cases
    # 1. All labels are the same
    if len(set(labels)) == 1:
        return Node(vote=labels[0], neg_count=neg_count, pos_count=pos_count)
    
    # 2. Max depth reached
    if max_depth is not None and current_depth >= max_depth:
        vote = majority_label(labels)
        return Node(vote=vote, neg_count=neg_count, pos_count=pos_count)
    
    # 3. Not enough data to split
    if len(data) < min_split_size:
        vote = majority_label(labels)
        return Node(vote=vote, neg_count=neg_count, pos_count=pos_count)
    
    # Find best attribute to split on
    num_attrs = len(data[0]) - 1  # exclude label column
    best_attr = None
    best_info_gain = -1
    
    for attr_idx in range(num_attrs):
        attr_values = [row[attr_idx] for row in data]
        info_gain = mutual_information(attr_values, labels)
        
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_attr = attr_idx
    
    # 4. Information gain below threshold
    if best_info_gain <= info_threshold:
        vote = majority_label(labels)
        return Node(vote=vote, neg_count=neg_count, pos_count=pos_count)
    
    # Create internal node
    node = Node(attr=best_attr, neg_count=neg_count, pos_count=pos_count)
    
    # Split data and create children
    left_data = split_data(data, best_attr, 0)
    right_data = split_data(data, best_attr, 1)
    
    node.left = build_tree(left_data, attr_names, max_depth, info_threshold, min_split_size, current_depth + 1)
    node.right = build_tree(right_data, attr_names, max_depth, info_threshold, min_split_size, current_depth + 1)
    
    return node

# Predict label for single example
def predict_single(tree, example):
    if tree.is_leaf():
        return tree.vote
    
    attr_value = example[tree.attr]
    if attr_value == 0:
        return predict_single(tree.left, example)
    else:
        return predict_single(tree.right, example)

# Predict labels for dataset
def predict(tree, data):
    return [predict_single(tree, row[:-1]) for row in data]

# Calculate accuracy
def calculate_accuracy(true_labels, predicted_labels):
    if len(true_labels) == 0:
        return 0.0
    correct = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p) # Count correct predictions
    return correct / len(true_labels)

# Sweep over depths and plot accuracies
""" Sweep max depth from 0..8 and plot training and validation accuracies. """
def depth_sweep(train_data, val_data, headers, depths=range(0, 9)):
    train_labels = [r[-1] for r in train_data]
    val_labels = [r[-1] for r in val_data]
    train_accs = []
    val_accs = []
    best_val = -1.0
    best_depth = None
    for d in depths:
        dt = DecisionTree(max_depth=d)
        dt.fit(train_data, headers)
        tr_preds = dt.predict(train_data)
        va_preds = dt.predict(val_data)
        tr_acc = calculate_accuracy(train_labels, tr_preds)
        va_acc = calculate_accuracy(val_labels, va_preds)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)
        if va_acc > best_val:
            best_val = va_acc
            best_depth = d
        print(f"Depth {d}: train_acc={tr_acc:.4f} val_acc={va_acc:.4f}")
    print(f"Best validation accuracy {best_val:.4f} at depth {best_depth}")
    if plt:
        plt.figure(figsize=(6,4))
        plt.plot(list(depths), train_accs, marker='o', label='Train')
        plt.plot(list(depths), val_accs, marker='s', label='Validation')
        plt.xlabel('Max Depth')
        plt.ylabel('Accuracy')
        plt.title('Depth vs Accuracy')
        plt.xticks(list(depths))
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("matplotlib not available; skipping plot.")

# Sweep over minimum split sizes and plot accuracies
""" Sweep minimum split size C from 0..N (N = number of training examples). """
def min_split_sweep(train_data, val_data, headers, max_depth=None):
    N = len(train_data)
    val_labels = [r[-1] for r in val_data]
    results = []
    best_acc = -1.0
    best_C = None
    for C in range(0, N + 1):
        dt = DecisionTree(max_depth=max_depth, min_split_size=C)
        dt.fit(train_data, headers)
        val_preds = dt.predict(val_data)
        acc = calculate_accuracy(val_labels, val_preds)
        results.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_C = C
        # If tie, keep smallest C
        print(f"C={C}: val_acc={acc:.4f}")
    print(f"Best validation accuracy {best_acc:.4f} at minimum split size C={best_C}")
    if plt:
        xs = list(range(0, N + 1))
        plt.figure(figsize=(7,4))
        plt.plot(xs, results, marker='.', linewidth=1)
        plt.xlabel('Minimum Split Size (C)')
        plt.ylabel('Validation Accuracy')
        plt.title('Min Split Size Sweep')
        plt.tight_layout()
        plt.show()
    else:
        print("matplotlib not available; skipping plot.")

# Sweep over information gain thresholds and plot accuracies
""" Sweep information gain threshold tau from 0.0 to 1.0 in increments of 0.01. """
def info_threshold_sweep(train_data, val_data, headers, max_depth=None, min_split_size=0):
    train_labels = [r[-1] for r in train_data]
    val_labels = [r[-1] for r in val_data]
    taus = [i / 100.0 for i in range(0, 101)] #Taus means thresholds from 0.0 to 1.0
    val_accs = []
    best_acc = -1.0
    best_tau = None
    for tau in taus:
        dt = DecisionTree(max_depth=max_depth, info_threshold=tau, min_split_size=min_split_size)
        dt.fit(train_data, headers)
        val_preds = dt.predict(val_data)
        acc = calculate_accuracy(val_labels, val_preds)
        val_accs.append(acc)
        if acc > best_acc or (acc == best_acc and (best_tau is None or tau < best_tau)):
            best_acc = acc
            best_tau = tau
        print(f"tau={tau:.2f}: val_acc={acc:.4f}")
    print(f"Best validation accuracy {best_acc:.4f} at tau={best_tau:.2f}")
    if plt:
        plt.figure(figsize=(7,4))
        plt.plot(taus, val_accs, marker='.', linewidth=1)
        plt.xlabel('Mutual Information Threshold (tau)')
        plt.ylabel('Validation Accuracy')
        plt.title('Information Gain Threshold Sweep')
        plt.tight_layout()
        plt.show()
    else:
        print("matplotlib not available; skipping plot.")

# Print decision tree
""" Print the decision tree in a readable format. """
def print_tree(tree, attr_names, depth=0, parent_attr=None, attr_value=None):
    indent = "| " * depth
    
    if depth == 0:
        # Root node
        print(f"[{tree.neg_count} 0/{tree.pos_count} 1]")
    else:
        # Non-root node
        print(f"{indent}{parent_attr} = {attr_value}: [{tree.neg_count} 0/{tree.pos_count} 1]")
    
    if not tree.is_leaf():
        # Print children
        print_tree(tree.left, attr_names, depth + 1, attr_names[tree.attr], 0)
        print_tree(tree.right, attr_names, depth + 1, attr_names[tree.attr], 1)

# Prune the tree using validation data
"""Prune tree means to reduce the size of the tree by removing nodes that do not provide power to classify instances"""
def prune_tree(tree, val_data, attr_names):
    # Reduced error pruning using only validation examples that reach each node
    if tree.is_leaf():
        return tree

    # If no validation examples reach this node, cannot evaluate pruning: keep subtree
    if not val_data:
        return tree

    # Split validation data according to this node's attribute
    attr_idx = tree.attr
    left_val = [row for row in val_data if row[attr_idx] == 0]
    right_val = [row for row in val_data if row[attr_idx] == 1]

    # Recurse
    tree.left = prune_tree(tree.left, left_val, attr_names)
    tree.right = prune_tree(tree.right, right_val, attr_names)

    # Evaluate accuracy before pruning (on the validation examples that reach this node)
    val_labels = [row[-1] for row in val_data]
    predictions_before = [predict_single(tree, row[:-1]) for row in val_data]
    acc_before = calculate_accuracy(val_labels, predictions_before)

    # Candidate leaf: majority label of this node's validation subset
    leaf_vote = majority_label(val_labels)

    # Save original
    original_attr = tree.attr
    original_left = tree.left
    original_right = tree.right
    original_vote = tree.vote

    # Temporarily prune
    tree.attr = None
    tree.left = None
    tree.right = None
    tree.vote = leaf_vote

    predictions_after = [predict_single(tree, row[:-1]) for row in val_data]
    acc_after = calculate_accuracy(val_labels, predictions_after)

    # Keep pruning if accuracy not worse (tie -> prefer simpler tree)
    if acc_after >= acc_before:
        return tree
    else:
        # Revert
        tree.attr = original_attr
        tree.left = original_left
        tree.right = original_right
        tree.vote = original_vote
        return tree

# Decision Tree Classifier
""" Decision Tree Classifier with fit, predict, print, and prune methods. """
class DecisionTree:
    
    def __init__(self, max_depth=None, info_threshold=0.0, min_split_size=0):
        self.max_depth = max_depth
        self.info_threshold = info_threshold
        self.min_split_size = min_split_size
        self.tree = None
        self.attr_names = None

    # Train the decision tree"""
    def fit(self, train_data, attr_names):
        self.attr_names = attr_names[:-1]  # exclude label column
        self.tree = build_tree(train_data, self.attr_names, 
                              self.max_depth, self.info_threshold, self.min_split_size)
    
    # Predict labels
    def predict(self, data):
        return predict(self.tree, data)
    
    # Print the decision tree
    def print_tree(self):
        if self.tree:
            print_tree(self.tree, self.attr_names)
    
    # Prune the tree using validation data
    """ prune means to reduce the size of the tree by removing nodes that do not provide power to classify instances """
    def prune(self, val_data):
        if self.tree:
            self.tree = prune_tree(self.tree, val_data, self.attr_names)


# Main function to run the decision tree
def main():
    if len(sys.argv) != 4:
        print("Usage: python decision_tree.py <train_file> <val_file> <max_depth>")
        sys.exit(1)
    
    # Parse command line arguments
    train_file = sys.argv[1]
    val_file = sys.argv[2]
    depth_arg = sys.argv[3]
    
    # Load data
    train_headers, train_data = load_data(train_file)
    val_headers, val_data = load_data(val_file)
    
    # depth sweep if specified
    if depth_arg.lower() == 'sweep':
        depth_sweep(train_data, val_data, train_headers)
        return
    
    # min split sweep if specified
    if depth_arg.lower() == 'minsplitsweep':
        # By default allow unlimited depth (None). Adjust if you want a cap.
        min_split_sweep(train_data, val_data, train_headers, max_depth=None)
        return
    
    # info threshold sweep if specified
    if depth_arg.lower() == 'infothresholdsweep':
        # By default allow unlimited depth (None) and min split size 0. Adjust if you want different values.
        info_threshold_sweep(train_data, val_data, train_headers, max_depth=None, min_split_size=0)
        return
    
    # Set max depth
    max_depth = int(depth_arg)

    # Build and train tree
    dt = DecisionTree(max_depth=max_depth)
    dt.fit(train_data, train_headers)
    
    # Prune tree
    dt.prune(val_data)

    # Print tree
    dt.print_tree()


if __name__ == '__main__':
    main()
