from math import log2
import numpy as np
from structures import Node

# Split condition, outputs X s.t. X1 = X1>=mean and X2<mean
def split_by_mean(X, mean):
    return (X[X>=mean], X[X<mean])

# NEED TO FIX --> CONDITIONAL ENTROPY
def entropy(X, condition):
    # Condition will always be binary
    count_yes = 0
    count_no = 0
    N = len(X)
    for x in X:
        if condition(x):
            count_yes += 1
        else:
            count_no += 1
    P_x = count_yes/N
    P_not_x = count_no/N
    yes_case = P_x*log2(P_x)
    no_case = P_not_x*log2(P_not_x)
    H = -(yes_case+no_case)
    return H

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
    for col in range(cols):
        extracted = X[:,col]
        # Split data based on mean
        mean = np.mean(extracted)
        x1, x2 = split_by_mean(extracted, X)
        
    # NEED TO FIND IG FOR EACH FEATURE
    # ASSUMING 'condition' IS THE NEW CONDITION
    condition = ...
    newNode = Node()
    newNode.add_condition(condition)
    # SPLIT INTO X1, X2, Y1, Y2 BASED ON CONDITION
    yesNode = learn_Node(Node(), X1, Y1)
    noNode = learn_Node(Node(), X2, Y2)
    newNode.set_children(noNode, yesNode)
    return newNode
    
def learn(X, Y, impurity_measure):
    if impurity_measure == 'entropy':
        metric = entropy()
    rootNode = Node()
    rootNode = learn_Node(rootNode, X, Y, metric)
    return rootNode
    # Can then use rootNode.predict(x) to predict any x
        
    