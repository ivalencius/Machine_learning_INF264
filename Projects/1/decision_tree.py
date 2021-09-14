from math import log2
from tokenize import String
import numpy as np
from numpy.lib.npyio import genfromtxt
from structures import Node

# Split condition, outputs X s.t. X1 = X1>=mean and X2<mean
def split_by_mean(X, Y, mean, col):
    idx_yes = X[:, col]>=mean
    idx_no = X[:, col]<mean
    return (X[idx_yes], X[idx_no], Y[idx_yes], Y[idx_no])

# NEED TO FIX --> CONDITIONAL ENTROPY
def entropy(Y, Y_split1, Y_split2, conditional):
    # Condition will always be binary
    # count_yes = 0
    # count_no = 0
    # N = len(X)
    # for x in X:
    #     if condition(x):
    #         count_yes += 1
    #     else:
    #         count_no += 1
    # P_x = count_yes/N
    # P_not_x = count_no/N
    # yes_case = P_x*log2(P_x)
    # no_case = P_not_x*log2(P_not_x)
    # H = -(yes_case+no_case)
    N = len(Y)
    (unique, counts) = np.unique(Y, return_counts=True)
    H = 0
    if not conditional:
        for xi, xn in zip(unique, counts):
            Hi= xi/N * log2(xi/N)
            H += Hi
        H = -H
    if conditional:
        H = 0
        
        split1_H = entropy(Y_split1, Y_split1, Y_split2, conditional=False)
        split2_H = entropy(Y_split2, Y_split1, Y_split2, conditional=False)
        split1_N = len(Y_split1)
        split2_N = len(Y_split2)
        
        H = (split1_N/N)*split1_H + (split2_N/N)*split2_H
        # for xi, xn in zip(unique, counts):
        #     # split_xi = len(Y_split1[Y_split1==xi])
        #     # split_N = len(Y_split1)
            
        #     H += xi/N * -(split_xi/split_N * log2(split_xi/split_N))     
    return H

def information_gain(Y, Y_split1, Y_split2, metric):
    if metric == 0: # Use Entropy
        H1 = entropy(Y, Y_split1, Y_split2, conditional=False)
        H2 = entropy(Y, Y_split1, Y_split2, conditional=True)
    return H1-H2
    

def learn_Node(node, X, Y, metric):
    # Case 1:
    # If all data points have same label
    labels, counts= np.unique(Y, return_counts=True)
    if len(labels) == 1:
        # print('Case 1')
        newLeaf = Node()
        newLeaf.make_leaf(label=Y.item(0))
        # print(newLeaf)
        return newLeaf
    
    # Case 2:
    # If all data points have identical feature values
    unique_feature = len(np.unique(X))
    if unique_feature == 1:
        # print('Case 2')
        label_idx = counts.index(max((counts)))
        newLeaf = Node()
        newLeaf.make_leaf(label=labels[label_idx])
        # print(newLeaf)
        return newLeaf
    
    # Case 3:
    # Get Entropy for each feature
    cols = X.shape[1]
    IGs = [] # Index = feature, will store (IG, mean)
    Means = []
    for c in range(cols):
        # Split data based on mean
        mean = np.mean(X[:, c])
        _, _, y_split1, y_split2 = split_by_mean(X, Y, mean, col=c)
        IGs.append(information_gain(Y, y_split1, y_split2, metric))
        Means.append(mean)

    # Determine feature with max gain
    # feature_idx = list.index(max(IGs))
    feature_idx = np.argmax(IGs, axis=0) 
    # Remember this index also index of the FEATURE
    condition = lambda x: x[feature_idx] >= Means[feature_idx]
    node.add_condition(condition)
    # SPLIT INTO X1, X2, Y1, Y2 BASED ON CONDITION
    X1, X2, Y1, Y2 = split_by_mean(X, Y, Means[feature_idx], col=feature_idx)
    yesNode = Node()
    noNode= Node()
    node_left = learn_Node(noNode, X2, Y2, metric)
    node_right = learn_Node(yesNode, X1, Y1, metric)
    node.set_children(node_left, node_right)
    # print('Set Children')
    return node
    
def learn(root, X, Y, impurity_measure):
    if impurity_measure == 'entropy':
        metric = 0
    learn_Node(root, X, Y, metric)
    # return rootNode
    # Can then use rootNode.predict(x) to predict any x

def load_magic(filename):
    X = []
    Y = []
    for line in filename:
        processed = line.split(',')
        x_line = list(map(float, processed[0:9]))
        y_label = processed[-1][0]
        y_int = ord(y_label)
        X.append(x_line)
        Y.append(y_int)
    # print(x_line)
    # print(y_line)
    return np.asarray(X), np.asarray(Y)

def main():
    magic04 = open('magic04.data', 'r')
    # D = np.genfromtxt(magic04, delimiter=',')
    # print('Data shape: '+str(D.shape))
    # X = D[:, 0:9]
    # Y = D[:, 10]
    print('\n** LOADING DATA **')
    X, Y = load_magic(magic04)
    print('Shape of X: '+str(X.shape))
    print('Shape of Y: '+str(Y.shape))
    
    print('\n** TRAINING **')
    # Y labels are now in ASCII --> convert back to determine class
    Tree = Node()
    learn(Tree, X, Y, impurity_measure='entropy')
    
    print('\n** PRINT TREE **')
    Tree.print_Nodes()
    
    print('\n** EVALUATING **')
    # Y_pred = []
    row, _ = X.shape
    correct = 0
    for i in range(row):
        # print('\nPrediction '+str(i)+': ' +Tree.predict(X[i,:]))
        # Y_pred.append(ord(X[i, :]))
        pred = ord(Tree.predict(X[i, :]))
        if pred == Y[i]: correct += 1
    
    print('\n** ACCURACY **')
    acc = correct/row
    print('\t->'+str(acc))
        
    
main()