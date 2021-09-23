from math import log2
import numpy as np
from structures import Node
from sklearn.model_selection import train_test_split
from sklearn import tree
from timeit import default_timer as timer

def split_by_mean(X, Y, mean, col):
    """ Splits X and Y

    Args:
        X (ndarray): features
        Y (ndarray): labels
        mean (float): value to split on
        col (int): column of feature which is split

    Returns:
        (ndarray, ndarray, ndarray, ndarray): X1, X2, Y1, Y2 s.t for all x in X1, Y1, x[col] >= mean
    """
    rows, _ = X.shape
    idx_yes = []
    idx_no = []
    for r in range(rows):
        if X[r, col] >= mean:
            idx_yes.append(r)
        else:
            idx_no.append(r)
    return (X[idx_yes], X[idx_no], Y[idx_yes], Y[idx_no])


def entropy(Y, Y_split1, Y_split2, conditional):
    """ Determines the entropy

    Args:
        Y (ndarray): Array before split
        Y_split1 (ndarray): subset of Y
        Y_split2 (ndarray):  other subset of Y
        conditional (bool): if entropy is determined conditionally 

    Returns:
        float: entropy
    """
    N = len(Y)
    _, counts = np.unique(Y, return_counts=True)
    H = 0
    if not conditional:
        for xn in counts:
            Hi= xn/N * log2(xn/N)
            H += Hi
        H = -H
    if conditional:
        H = 0
        
        # Determine entropy of right and left branches
        split1_H = entropy(Y_split1, Y_split1, Y_split2, conditional=False)
        split2_H = entropy(Y_split2, Y_split1, Y_split2, conditional=False)
        split1_N = len(Y_split1)
        split2_N = len(Y_split2)
        
        H = (split1_N/N)*split1_H + (split2_N/N)*split2_H
 
    return H

def gini(Y):
    """ Determines the gini index of an array Y

    Args:
        Y (ndarray): array of labels

    Returns:
        float: gini_index
    """
    N = len(Y)
    _, counts = np.unique(Y, return_counts=True)
    G = 0
    for xn in counts:
        Gi = (xn/N) **2
        G += Gi
    return 1-G

        
def determine_metric(Y, Y_split1, Y_split2, metric):
    """ Determines the value of the determined metric on a node

    Args:
        Y (ndarray): full data labels
        Y_split1 (ndarray): one subset of Y
        Y_split2 (ndarray): other subset of Y
        metric (int): determines which metric to use

    Returns:
        float: information gain or gini index of a node
    """
    if metric == 0: # Use Entropy
        H1 = entropy(Y, Y_split1, Y_split2, conditional=False)
        H2 = entropy(Y, Y_split1, Y_split2, conditional=True)
        IG = H1-H2
    elif metric == 1: # Use Gini
        if len(Y) == 0:
            return 0
        w1 = len(Y_split1)/len(Y)
        w2 = len(Y_split2)/len(Y)
        IG = gini(Y) - (gini(Y_split1)*w1 + gini(Y_split2)*w2)
        
    return IG
    

def learn_Node(node, X, Y, metric):
    """Learns a node based on X, Y, metric

    Args:
        node (Node()): node to be
        X (ndarray): feature of
        Y (ndarray): labels
        metric (int): determines gini or entropy

    Returns:
        Node(): node with children learned as well
    """
    # Case 1:
    # If all data points have same label
    labels, counts= np.unique(Y, return_counts=True)
    if len(labels) == 1:
        newLeaf = Node()
        newLeaf.make_leaf(label=Y.item(0))
        # if type(Y.item(0)) != int:
        #     print(Y.item(0))
        return newLeaf

    # Case 2:
    # If all data points have identical feature values
    if len(np.unique(X)) == 1:
        label_idx = counts.index(max((counts)))
        newLeaf = Node()
        newLeaf.make_leaf(label=labels[label_idx])
        # print(labels[label_idx])
        return newLeaf
    
    # Case 3:
    # Get metric for each feature
    cols = X.shape[1]
    metric_vals = [] # Index = feature, will store (IG, mean)
    Means = []
    for c in range(cols):
        # Split data based on mean
        mean = np.mean(X[:, c])
        _, _, y_split1, y_split2 = split_by_mean(X, Y, mean, col=c)
        metric_vals.append(determine_metric(Y, y_split1, y_split2, metric))
        # print(len(y_split1))
        Means.append(mean)

    # Determine feature with max gain
    feature_idx = np.argmax(metric_vals, axis=0) # Remember this index also index of the FEATURE
    
    node.add_condition(feature_idx, Means[feature_idx])
    
    # SPLIT INTO X1, X2, Y1, Y2 BASED ON CONDITION
    X1, Y1, X2, Y2 = split(node, X, Y)
    
    yesNode = Node()
    noNode= Node()
    
    node_left = learn_Node(noNode, X2, Y2, metric)
    node_right = learn_Node(yesNode, X1, Y1, metric)
    
    node.set_children(node_left, node_right)
    return node


def split(node, X, Y):
    """Splits X, Y, based on condition of a node

    Args:
        node (Node()): determines how data is split
        X (ndarray): feature values
        Y (ndarray): labels

    Returns:
        (ndarray, ndarray, ndarray, ndarray): X1, Y1, X2, Y2 s.t. X1, Y1 classified as true under condition of node
    """
    data_right = []
    labels_right = []
    data_left = []
    labels_left =[]
    try:
        row, _ = X.shape
    except:
        row = 0
    for i in range(row):
        if node.test_condition(X[i, :]):
            data_right.append(X[i, :])
            labels_right.append(Y[i])
        else:
            data_left.append(X[i, :])
            labels_left.append(Y[i])
    return np.array(data_right), np.array(labels_right), np.array(data_left), np.array(labels_left)

def prune(node, X_prune, Y_prune, X_train, Y_train):
    """Prunes a Node() node

    Args:
        node (Node()): node to be pruned
        X_prune (ndarray): pruning features
        Y_prune (ndarray): pruning labels
        X_train (ndarray): training features
        Y_train (ndarray): training labels

    Returns:
        Node(): pruned node with children pruned
    """
    if node.is_leaf():
        return node
    
    if len(Y_prune) == 0:
        return node
    
    # Split training and pruning data based on condition of node
    X_prune_r, Y_prune_r, X_prune_l, Y_prune_l = split(node, X_prune, Y_prune)
    X_train_r, Y_train_r, X_train_l, Y_train_l = split(node, X_train, Y_train)

    
    # Prune children nodes
    prune(node.right, X_prune_r, Y_prune_r, X_train_r, Y_train_r)
    prune(node.left, X_prune_l, Y_prune_l, X_train_l, Y_train_l)
    
    # 1 - Test node without split
    labels, counts = np.unique(Y_train, return_counts=True)
    
    try:
        maj_label_arr = labels[np.where(counts == np.ndarray.max(counts))]
        maj_label = maj_label_arr[0]

    except:
        maj_label = labels # case where labels not an array -> just one value
    
    # 2 - Error with no split
    wrong_count = 0
    for i in range(len(Y_prune)):
        if Y_prune[i] != maj_label: 
            wrong_count += 1
    err_no_split = wrong_count/len(Y_prune)
    
    # 3- Error with split
    
    err_split = 1- get_acc(node, X_prune, Y_prune)
    
    if err_no_split <= err_split:
        node.clear_node()
        node.make_leaf(maj_label)
        
    return node
    
def learn(root, X, Y, impurity_measure, pruning=False, prune_sz=0, seed=None):
    """Trains a root node

    Args:
        root (Node()): root node to be trained
        X (ndarray): training features
        Y (ndarray): training labels
        impurity_measure (int): determines impurity measure
        pruning (bool, optional): determines whether to prune. Defaults to False.
        prune_sz (int, optional): size of prune dataset. Defaults to 0.
        seed (int, optional): sets seed for splitting prune data. Defaults to None.

    Returns:
        [type]: [description]
    """
    if pruning:
        # Divide into pruning data
        X, X_prune, Y, Y_prune = train_test_split(X,
                                                Y,
                                                test_size=prune_sz,
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
    """Loads data from MAGIC Gamma Telescope 

    Args:
        filename (file object): file to parse

    Returns:
        (ndarray, ndarray): X, Y s.t. X is features and Y is labels
    """
    X = []
    Y = []
    for line in filename:
        processed = line.split(',')
        x_line = list(map(float, processed[0:9]))
        y_label = processed[-1][0]
        y_int = ord(y_label)
        X.append(x_line)
        Y.append(y_int)
    return np.asarray(X), np.asarray(Y)

def get_acc(node, X, Y):
    row, _ = X.shape
    correct = 0
    for i in range(row):
        pred = node.predict(X[i, :])
        if pred == chr(Y[i]): correct += 1
        
    return correct/len(Y)

def main():
    """Organizes function calls for training and evaluating decision tree models
    """
    
    print('\n** LOADING DATA **')
    
    magic04 = open('magic04.data', 'r')
    X, Y = load_magic(magic04)

    seed = None
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X,
                                                        Y,
                                                        test_size=0.15,
                                                        shuffle=True, 
                                                        random_state=seed)
    
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val,
                                                      Y_train_val, 
                                                      test_size=0.15,
                                                      shuffle=True,
                                                      random_state=seed)

    print('\n** TRAINING **')

        
    Tree_e =  Node()
    Tree_ep = Node()
    Tree_g = Node()
    Tree_gp = Node() 
    Trees = [Tree_e, Tree_ep, Tree_g, Tree_gp]
    params = [('entropy', False, 0.0),
              ('entropy', True, 0.3),
              ('gini', False, 0.0),
              ('gini', True, 0.3)]
    start_all = timer()
    for Tree, param in zip(Trees, params): 
        metric, prune, sz = param
        learn(Tree,
              X_train,
              Y_train,
              impurity_measure=metric,
              pruning=prune,
              prune_sz=sz,
              seed=seed)
    end_all = timer()
    avg_time = (end_all-start_all)/4
    
    
    # print('\n** PRINT TREE **')
    # # Tree.print_Nodes()
    
    print('\n** EVALUATING **')
    accs = []
    for Tree, param in zip(Trees, params): 
        acc = str(get_acc(Tree, X_val, Y_val))
        accs.append(acc)
        print('\n\tTREE:')
        print('\tMetric: %s | Pruning: %s | Prune Size: %f'%param)
        print('\tValidation Accuracy: '+acc)    
        
    print('\n** BEST MODEL **')
    best = accs.index(max(accs))
    best_param = params[best]
    best_acc = str(get_acc(Trees[best], X_test, Y_test))
    # print('TREE:')
    print('\tMetric: %s | Pruning: %s | Prune Size: %f'%best_param)
    print('\tAccuracy on Test Set: '+best_acc)
    
    # Test on sklearn
    sk_tree = tree.DecisionTreeClassifier()
    
    start_sk = timer()
    sk_tree.fit(X_train, Y_train)
    end_sk = timer()
    
    sk_time = end_sk-start_sk
    
    preds = sk_tree.predict(X_test)
    
    correct = 0
    for pred, y in zip(preds, Y_test):
        if pred == y:
            correct += 1
    
    print('\n** SKLEARN **')
    print('\tAccuracy on Test Set: '+str(correct/len(preds)))
    
    print('\n** TIMING **')
    print('\tAverage homebrew training time: %f'%(avg_time))
    print('\tSklearn training time: %f'%(sk_time))
    
main()