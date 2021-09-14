from math import log2
from tokenize import String
import numpy as np
from numpy.lib.npyio import genfromtxt
from structures import Node

# Split condition, outputs X s.t. X1 = X1>=mean and X2<mean
def split_by_mean(X, Y, mean):
    idx_yes = X>=mean
    idx_no = X<mean
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
        
        split1_H = entropy(Y_split1, conditional=False)
        split2_H = entropy(Y_split2, conditional=False)
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
        H1 = entropy(Y, conditional=False)
        H2 = entropy(Y, Y_split1, Y_split2, conditional=True)
    return H1-H2
    

def learn_Node(node, X, Y, metric):
    # Case 1:
    # If all data points have same label
    labels, counts = len(np.unique(Y))
    if len(labels) == 1:
        newLeaf = Node()
        newLeaf.make_leaf(label=Y.item(0))
        return newLeaf
    
    # Case 2:
    # If all data points have identical feature values
    unique_feature = len(np.unique(X))
    if unique_feature == 1:
        label_idx = counts.index(max((counts)))
        newLeaf = Node()
        newLeaf.make_leaf(label=labels[label_idx])
    
    # Case 3:
    # Get Entropy for each feature
    cols = X.shape[1]
    IGs = [] # Index = feature, will store (IG, mean)
    Means = []
    for col in range(cols):
        extracted = X[:,col]
        # Split data based on mean
        mean = np.mean(extracted)
        _, _, y_split1, y_split2 = split_by_mean(extracted, Y, mean)
        IGs.append(information_gain(Y, y_split1, y_split2, metric))
        Means.append(mean)
    
    # Determine feature with max gain
    feature_idx = list.index(max(IGs)) 
    # Remember this index also index of the FEATURE
    condition = lambda x: x[:, feature_idx] >= Means[feature_idx]
    node.add_condition(condition)
    # SPLIT INTO X1, X2, Y1, Y2 BASED ON CONDITION
    X1, X2, Y1, Y2 = split_by_mean(X[:, feature_idx], Y, Means[feature_idx])
    yesNode = learn_Node(Node(), X1, Y1)
    noNode = learn_Node(Node(), X2, Y2)
    node.set_children(noNode, yesNode)
    return node
    
def learn(X, Y, impurity_measure):
    if impurity_measure == 'entropy':
        metric = 0
    rootNode = Node()
    rootNode = learn_Node(rootNode, X, Y, metric)
    return rootNode
    # Can then use rootNode.predict(x) to predict any x

def load_magic(filename):
    X = []
    Y = []
    for line in filename:
        processed = line.split(',')
        x_line = list(map(float, processed[0:9]))
        y_label = processed[-1][0]
        y_int = int(float(y_label))
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
    X, Y = load_magic(magic04)
    print('Shape of X: '+str(X.shape))
    print('Shape of Y: '+str(Y.shape))
    # Y labels are now ints --> convert back to determine class
    Tree = learn(X, Y, impurity_measure='entropy')
    
    Tree.predict(X[1,:])
main()