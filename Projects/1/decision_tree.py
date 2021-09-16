from math import log2
import numpy as np
from numpy.lib.npyio import genfromtxt
from structures import Node
from sklearn.model_selection import train_test_split

# Split condition, outputs X s.t. X1 = X1>=mean and X2<mean
def split_by_mean(X, Y, mean, col):
    rows, _ = X.shape
    idx_yes = []
    idx_no = []
    for r in range(rows):
        if X[r, col] >= mean:
            idx_yes.append(r)
        else:
            idx_no.append(r)
    # idx_yes = X[:, col]>=mean
    # idx_no = X[:, col]<mean
    # print('X_idx:'+str(len(idx_yes)))
    # print('X subset:'+str(len(X[idx_yes])))
    return (X[idx_yes], X[idx_no], Y[idx_yes], Y[idx_no])

def entropy(Y, Y_split1, Y_split2, conditional):
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
def gini(Y):
    N = len(Y)
    (unique, counts) = np.unique(Y, return_counts=True)
    G = 0
    for xi, xn in zip(unique, counts):
        Gi= xi/N * (1-(xi/N))
        G += Gi
    return G
        
def information_gain(Y, Y_split1, Y_split2, metric):
    if metric == 0: # Use Entropy
        H1 = entropy(Y, Y_split1, Y_split2, conditional=False)
        H2 = entropy(Y, Y_split1, Y_split2, conditional=True)
        IG = H1-H2
    elif metric == 1: # Use Gini
        IG = gini(Y_split1)-gini(Y_split2)
    return IG
    

def learn_Node(node, X, Y, metric):
    # Case 1:
    # If all data points have same label
    labels, counts= np.unique(Y, return_counts=True)
    if len(labels) == 1:
        # print(Y)
        # print()
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
        # print(len(y_split1))
        Means.append(mean)

    # Determine feature with max gain
    # feature_idx = list.index(max(IGs))
    feature_idx = np.argmax(IGs, axis=0) 
    # Remember this index also index of the FEATURE
    condition = lambda x: x[feature_idx] >= Means[feature_idx]
    node.add_condition(condition)
    # SPLIT INTO X1, X2, Y1, Y2 BASED ON CONDITION
    X1, X2, Y1, Y2 = split_by_mean(X, Y, Means[feature_idx], col=feature_idx)
    # print('X len:'+str(len(X)))
    # print('X1:'+str(len(X1)))
    # print('X2:'+str(len(X2)))
    # print(Y1)
    # print(Y2)
    # print()
    yesNode = Node()
    noNode= Node()
    node_left = learn_Node(noNode, X2, Y2, metric)
    # node_left.print_Nodes()
    node_right = learn_Node(yesNode, X1, Y1, metric)
    # node_right.print_Nodes()
    node.set_children(node_left, node_right)
    # print('Set Children')
    # node.print_Nodes()
    # print(node.left)
    return node
    
def learn(root, X, Y, impurity_measure):
    if impurity_measure == 'entropy':
        metric = 0
    elif impurity_measure == 'gini':
        metric = 1
    else:
        print('ERROR: IMPROPER LEARNING METRIC')
        return
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
    print('\n** LOADING DATA **')
    magic04 = open('magic04.data', 'r')
    # D = np.genfromtxt(magic04, delimiter=',')
    # print('Data shape: '+str(D.shape))
    # X = D[:, 0:9]
    # Y = D[:, 10]
    X, Y = load_magic(magic04)
    # print('Shape of X: '+str(X.shape))
    # print('Shape of Y: '+str(Y.shape))
    seed = 156
    X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                        Y,
                                                        test_size=0.2,
                                                        shuffle=True, 
                                                        random_state=seed)
    
    
    print('\n** TRAINING **')
    # Y labels are now in ASCII --> convert back to determine class
    impurity = 'gini'
    entropy_tree = Node()
    learn(entropy_tree, X_train, Y_train, impurity_measure=impurity)
    print('Finished training %s model'%(impurity))
    
    print('\n** PRINT TREE **')
    # Tree.print_Nodes()
    
    print('\n** EVALUATING **')
    # Y_pred = []
    row, _ = X_test.shape
    correct = 0
    for i in range(row):
        # print('\nPrediction '+str(i)+': ' +Tree.predict(X[i,:]))
        # Y_pred.append(ord(X[i, :]))
        pred = ord(entropy_tree.predict(X_test[i, :]))
        if pred == Y_test[i]: correct += 1
    
    print('\n** ACCURACY **')
    acc = correct/row
    print('\t->'+str(acc))
        
    
main()