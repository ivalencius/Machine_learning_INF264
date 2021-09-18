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
        # node.set_weight(len(X))
        # print(newLeaf)
        return newLeaf
    # print('here')
    # Case 2:
    # If all data points have identical feature values
    unique_feature = len(np.unique(X))
    if unique_feature == 1:
        # print('Case 2')
        label_idx = counts.index(max((counts)))
        newLeaf = Node()
        newLeaf.make_leaf(label=labels[label_idx])
        # node.set_weight(len(X))
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

# Returns correct predictions of a node
# def predict_counts(node, X, Y):
#     count = 0
#     for x, y in zip(X, Y):
#         if node.predict(x):
#             count += 1
#     return count, len(Y)

def split(node, X, Y):
    data_right = []
    labels_right = []
    data_left = []
    labels_left =[]
    row, _ = X.shape
    for i in range(row):
        if node.predict(X[i, :]) == chr(Y[i]):
            data_right.append(X[i, :])
            labels_right.append(Y[i])
        else:
            data_left.append(X[i, :])
            labels_left.append(Y[i])
    return np.array(data_right), np.array(labels_right), np.array(data_left), np.array(labels_left)

# Determines accuracy of a subtree
# def get_child_counts(node, X, Y):
#     num = 0
#     count = 0

#     # If node is leaf
#     if node.get_label() != None:
#         label = node.get_label()
#         for y in Y:
#             if y == label:
#                 num += 1
#         count += len(Y)
#         return num, count
#     else:
#         # Remember right = classified as true
#         x_r, y_r, x_l, y_l = split(node, X, Y)
#         correct_r, num_r = get_child_counts(node.right,
#                                             x_r,
#                                             y_r)
#         correct_l, num_l = get_child_counts(node.left,
#                                             x_l,
#                                             y_l)
#     return (correct_r+correct_l), (num_r+num_l)
    
# def get_majority_label(node, dict={}):
#     if node.get_label() is not None:
#         # print(node.get_label())
#         try:
#             dict[node.label] += node.get_weight()
#         except:
#             # try:
#             # print(node.label)
#             dict[node.label] = node.get_weight()
#             # except:
#             #     dict = {node.label : 1}
#     else:
#         dict_2 = get_majority_label(node.right, dict)
#         dict_3 = get_majority_label(node.left, dict_2)    
#         # majority_label = max(dict_3, key=dict_3.get)
#         # print(dict_3)
#         # print(majority_label)
#         # return majority_label
#         return dict_3
    
#     return dict

    
# Returns a pointer to a node
# def prune(node, X_prune, Y_prune, X_train, Y_train):
#     # X_prune = np.asmatrix(X_prune)
#     #X_prune/Y_prune are now array of arrays
#     # Traversed to below a leaf
#     if node is None:
#         return node
    
#     # No data reached node
#     if len(X_prune) == 0:
#         return node # Empty pruning array
    
#     # If node is a leaf
#     if node.right and node.left is None:
#         return node # no pruning takes place --> leaf

#     # Remember right = classified as true
#     tx_r, ty_r, tx_l, ty_l = split(node, X_train, Y_train)
#     x_r, y_r, x_l, y_l = split(node, X_prune, Y_prune)
#     # Prune children first to start from bottom up
#     node.right = prune(node.right, x_r, y_r, tx_r, ty_r)
#     node.left = prune(node.left, x_l, y_l, tx_l, ty_l)

#     # correct, num = get_child_counts(node, X_prune, Y_prune)
#     # if num == 0:
#     #     return node # Case where no pruning data partitioned into node
#     # else:
#     #     acc_test = correct/num
#     # X_prune is an ndarray
#     # row,_ = X_prune.shape
    
#     # Accuracy with test
#     correct = 0
#     for i in range(len(X_prune)):
#         # print('\nPrediction '+str(i)+': ' +Tree.predict(X[i,:]))
#         # Y_pred.append(ord(X[i, :]))
#         pred = ord(node.predict(X_prune[i]))
#         if pred == Y_prune[i]: correct += 1
    
#     acc_test = correct/len(Y_prune)

#     # Accuracy with no test (based on majority labels)
#     # vals, counts = np.unique(Y_prune, return_counts=True)
#     # max_val_idx = np.where(vals == np.ndarray.max(vals))
#     # max_val = counts[max_val_idx]
#     # label_dict = get_majority_label(node)
#     # maj_label = max(label_dict, key=label_dict.get)
#     # # sum_label = sum(label_dict.values())
#     # maj_label_weight = label_dict[maj_label]
#     # maj_label = 0
#     acc_test = len(tx_r)/len(X_train) # -> # predicted correctly/len
#     # c = 0
#     # for y in Y_prune:
#     #     if y == maj_label: 
#     #         c+=1
#     # acc_no_test = c/len(Y_prune)
    
#     # Determine whether to prune
#     if acc_no_test >= acc_test:
#         # print('Acc no test: %f, | Acc test: %f'%(acc_no_test, acc_test))
#         # If pruning --> make leaf with majority label
#         node.set_children(None, None)
#         node.make_leaf(maj_label)
#         node.set_weight(maj_label_weight)
    
#     return node # Return pruned node

def prune(node, X_prune, Y_prune, X_train, Y_train):
    
    if node.is_leaf():
        return node
    
    if not node.has_children():
        return node
    
    # Split training and pruning data based on condition of node
    try:
        X_train_r, Y_train_r, X_train_l, Y_train_l = split(node, X_train, Y_train)
        X_prune_r, Y_prune_r, X_prune_l, Y_prune_l = split(node, X_prune, Y_prune)
    except:
        return node # No data to split
    # Prune children nodes
    node.right = prune(node.right, X_prune_r, Y_prune_r, X_train_r, Y_train_r)
    node.left = prune(node.left, X_prune_l, Y_prune_l, X_train_l, Y_train_l)
    
    # 1 - For each child node make a leaf with majority label
    # leaf_right = Node()
    # leaf_left = Node()
    # Get majority class
    labels, counts = np.unique(Y_train, return_counts=True)
    # maj_label = labels[counts.index(max(counts))]
    if type(labels) == np.int32:
        maj_label = labels
    else:
        maj_label = labels[np.where(counts == np.ndarray.max(counts))]
        maj_label = maj_label[0]
    print(maj_label)
    # labels_r, counts_r = np.unique(Y_train_r)
    # labels_l, counts_l = np.unique(Y_train_l)
    
    # maj_r = labels_r[counts_r.index(max(counts_r))]
    # maj_l = labels_l[counts_l.index(max(counts_l))]
    
    # # assign majority label to leaf
    # leaf_right.make_leaf(maj_r)
    # leaf_left.make_leaf(maj_l)
    
    # 2 - Accuracy with no split
    count = 0
    for i in range(len(Y_prune)):
        if Y_prune[i] == maj_label: 
            count += 1
    acc_no_split = count/len(Y_prune)
    
    # 3- Accuracy with split
    try:
        row, _ = X_prune.shape
    except:
        row = 1
    correct = 0
    for i in range(row):
        # print('\nPrediction '+str(i)+': ' +Tree.predict(X[i,:]))
        # Y_pred.append(ord(X[i, :]))
        pred = node.predict(X_prune[i, :])
        if pred == chr(Y_prune[i]): correct += 1
    acc_split = correct/len(Y_prune)
    
    if acc_no_split >= acc_split:
        print('Acc no split: %f, | Acc split: %f'%(acc_no_split, acc_split))
        # If pruning --> make leaf with majority label
        node.set_children(None, None)
        node.make_leaf(maj_label)
        
    return node
    
def learn(root, X, Y, impurity_measure, pruning=False, seed=None):
    if pruning:
        # Divide into pruning data
        X, X_prune, Y, Y_prune = train_test_split(X,
                                                  Y,
                                                  test_size=0.3,
                                                  shuffle=True,
                                                  random_state=seed)
    if impurity_measure == 'entropy':
        metric = 0
    elif impurity_measure == 'gini':
        metric = 1
    else:
        print('ERROR: IMPROPER LEARNING METRIC')
        return
    learn_Node(root, X, Y, metric)
    
    if pruning:
        prune(root, X_prune, Y_prune, X, Y)
    return root

def load_magic(filename):
    X = []
    Y = []
    # num_lines = sum(1 for line in filename)
    # X = np.empty((num_lines, 11))
    # Y = np.empty((num_lines, 1))
    # i = 0
    for line in filename:
        processed = line.split(',')
        x_line = list(map(float, processed[0:9]))
        y_label = processed[-1][0]
        y_int = ord(y_label)
        # X[i, :] = x_line
        # Y[i, 0] = y_int
        # i += 1
        X.append(x_line)
        Y.append(y_int)
    # print(x_line)
    # print(y_line)
    # X = np.asmatrix(X)
    # Y = np.asmatrix(Y)
    # Y = np.reshape((num_lines, -1))
    return np.asarray(X), np.asarray(Y)

def main():
    print('\n** LOADING DATA **')
    magic04 = open('magic04.data', 'r')
    # D = np.genfromtxt(magic04, delimiter=',')
    # print('Data shape: '+str(D.shape))
    # X = D[:, 0:9]
    # Y = D[:, 10]
    X, Y = load_magic(magic04)
    # X = np.asmatrix(X)
    # Y = np.asmatrix(Y)
    # Y labels are now in ASCII --> convert back to determine class
    # print('Shape of X: '+str(X.shape))
    # print('Shape of Y: '+str(Y.shape))
    seed = 156
    X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                        Y,
                                                        test_size=0.2,
                                                        shuffle=True, 
                                                        random_state=seed)

    print('\n** TRAINING **')
    impurity = 'gini'
    prune = True
    Tree = Node()
    learn(Tree, 
          X_train, 
          Y_train, 
          impurity_measure=impurity, 
          pruning=prune,
          seed=seed)
    
    print('Finished training %s model'%(impurity))
    
    print('\n** PRINT TREE **')
    Tree.print_Nodes()
    
    print('\n** EVALUATING **')
    # Y_pred = []
    row, _ = X_test.shape
    correct = 0
    for i in range(row):
        # print('\nPrediction '+str(i)+': ' +Tree.predict(X[i,:]))
        # Y_pred.append(ord(X[i, :]))
        pred = Tree.predict(X_test[i, :])
        if pred == chr(Y_test[i]): correct += 1
    
    print('\n** ACCURACY **')
    acc = correct/row
    print('\t->'+str(acc))
        
    
main()