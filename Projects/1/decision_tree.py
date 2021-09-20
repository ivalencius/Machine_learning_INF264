from math import log2
import numpy as np
from structures import Node
from sklearn.model_selection import train_test_split
from sklearn import tree

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
    # if N == 0:
    #     return 0
    N = len(Y)
    (_, counts) = np.unique(Y, return_counts=True)
    G = 0
    for xn in counts:
        Gi= xn/N * (1-(xn/N))
        G += Gi
    return G
    # Calculate percent branch represents -> weighting
    # w_Y1 = len(Y1)/N
    # w_Y2 = len(Y2)/N
    # # For each class in branch
    # # Branch 1
    # _, count_1 = np.unique(Y1, return_counts=True)
    # N1 = len(Y1)
    # sum_p1 = 0
    # for x in count_1:
    #     p_x = x/(N1)
    #     p_x2 = p_x**2
    #     sum_p1 += p_x2
    # gini_branch_1 = 1-sum_p1*w_Y1
    # # Branch 2
    # _, count_2 = np.unique(Y2, return_counts=True)
    # N2 = len(Y2)
    # sum_p2 = 0
    # for x in count_1:
    #     p_x = x/(N2)
    #     p_x2 = p_x**2
    #     sum_p2 += p_x2
    # gini_branch_2 = 1-sum_p2*w_Y2
    
    # return gini_branch_1+gini_branch_2
        
def information_gain(Y, Y_split1, Y_split2, metric):
    if metric == 0: # Use Entropy
        H1 = entropy(Y, Y_split1, Y_split2, conditional=False)
        H2 = entropy(Y, Y_split1, Y_split2, conditional=True)
        IG = H1-H2
    elif metric == 1: # Use Gini
        if len(Y) == 0:
            return 0
        w1 = len(Y_split1)/len(Y)
        w2 = len(Y_split2)/len(Y)
        IG = 1- (gini(Y_split1)*w1 + gini(Y_split2)*w2)
        # print(IG)
    return IG
    

def learn_Node(node, X, Y, metric):
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
    unique_feature = len(np.unique(X))
    if unique_feature == 1:
        label_idx = counts.index(max((counts)))
        newLeaf = Node()
        newLeaf.make_leaf(label=labels[label_idx])
        # print(labels[label_idx])
        return newLeaf
    
    # Case 3:
    # Get metric for each feature
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
    feature_idx = np.argmax(IGs, axis=0) 
    # Remember this index also index of the FEATURE
    # condition = lambda x: x[feature_idx] >= Means[feature_idx]
    node.add_condition(feature_idx, Means[feature_idx])
    # SPLIT INTO X1, X2, Y1, Y2 BASED ON CONDITION
    # X1, X2, Y1, Y2 = split_by_mean(X, Y, Means[feature_idx], col=feature_idx)
    X1, Y1, X2, Y2 = split(node, X, Y)
    
    yesNode = Node()
    noNode= Node()
    
    node_left = learn_Node(noNode, X2, Y2, metric)
    node_right = learn_Node(yesNode, X1, Y1, metric)
    
    node.set_children(node_left, node_right)
    return node


def split(node, X, Y):
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
    
    if node.is_leaf():
        return node
    
    if len(Y_prune) == 0:
        return node
    
    # if len(Y_train) == 0:
    #     return node
    
    # if not node.has_children():
    #     return node
    
    # if len(Y_train) == 0:
    #     return node
    
    # Split training and pruning data based on condition of node
    X_prune_r, Y_prune_r, X_prune_l, Y_prune_l = split(node, X_prune, Y_prune)
    X_train_r, Y_train_r, X_train_l, Y_train_l = split(node, X_train, Y_train)

    # return node # No data to split
    
    # Prune children nodes
    prune(node.right, X_prune_r, Y_prune_r, X_train_r, Y_train_r)
    prune(node.left, X_prune_l, Y_prune_l, X_train_l, Y_train_l)
    
    # 1 - Test node without split
    labels, counts = np.unique(Y_train, return_counts=True)
    # print(labels, counts)
    
    # maj_label = labels[counts.index(max(counts))]
    try:
        maj_label_arr = labels[np.where(counts == np.ndarray.max(counts))]
        maj_label = maj_label_arr[0]
        # print(labels, counts)
        # print('\t'+str(maj_label))
    except:
        maj_label = labels # case where labels not an array -> just one value
    # print(maj_label)
    
    # 2 - Accuracy with no split
    wrong_count = 0
    for i in range(len(Y_prune)):
        if Y_prune[i] != maj_label: 
            wrong_count += 1
    err_no_split = wrong_count/len(Y_prune)
    
    # 3- Accuracy with split
    try:
        row, _ = X_prune.shape
    except:
        row = 1 # Only one data point
        
    wrong = 0
    for i in range(row):
        pred = node.predict(X_prune[i, :])
        if pred != chr(Y_prune[i]): wrong += 1
        
    err_split = wrong/len(Y_prune)
    
    if err_no_split < err_split:
        # print('Err no split: %f, | Err split: %f'%(err_no_split, err_split))
        # If pruning --> make leaf with majority label
        node.clear_node()
        node.make_leaf(maj_label)
        # print(maj_label)
        
    return node
    
def learn(root, X, Y, impurity_measure, pruning=False, prune_sz=0, seed=None):
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
        print('\n** PRUNING **')
        prune(root, X_prune, Y_prune, X, Y)
    return root

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
    return np.asarray(X), np.asarray(Y)

def get_acc(node, X, Y):
    row, _ = X.shape
    correct = 0
    for i in range(row):
        # print('\nPrediction '+str(i)+': ' +Tree.predict(X[i,:]))
        # Y_pred.append(ord(X[i, :]))
        pred = node.predict(X[i, :])
        if pred == chr(Y[i]): correct += 1
    return correct/len(Y)

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
    
    # # For testing model
    # impurity = 'gini'
    # prune = True
    # Tree = Node()
    # learn(Tree, 
    #       X_train, 
    #       Y_train, 
    #       impurity_measure=impurity, 
    #       pruning=prune,
    #       prune_sz=0.3,
    #       seed=seed)
    # print('Finished training %s model'%(impurity))
    
    # print('\n** TREE **')
    
    # # Tree.print_Nodes()
    
    # print('\n** ACCURACY **\n'+str(get_acc(Tree, X_test, Y_test)))
        
    Tree_e =  Node()
    Tree_ep = Node()
    Tree_g = Node()
    Tree_gp = Node() 
    Trees = [Tree_e, Tree_ep, Tree_g, Tree_gp]
    params = [('entropy', False, 0.0),
              ('entropy', True, 0.3),
              ('gini', False, 0.0),
              ('gini', True, 0.3)]
    for Tree, param in zip(Trees, params): 
        metric, prune, sz = param
        learn(Tree,
              X_train,
              Y_train,
              impurity_measure=metric,
              pruning=prune,
              prune_sz=sz,
              seed=seed)
    
    
    print('\n** PRINT TREE **')
    # Tree.print_Nodes()
    
    print('\n** EVALUATING **')
    accs = []
    for Tree, param in zip(Trees, params): 
        acc = str(get_acc(Tree, X_val, Y_val))
        accs.append(acc)
        # print('TREE:')
        print('\tMetric: %s | Pruning: %d | Prune Size: %f'%param)
        print('\tValidation Accuracy: '+acc)    
        
    print('\n** BEST MODEL **')
    best = accs.index(max(accs))
    best_param = params[best]
    best_acc = str(get_acc(Trees[best], X_test, Y_test))
    # print('TREE:')
    print('\tMetric: %s | Pruning: %d | Prune Size: %f'%best_param)
    print('\tAccuracy on Test Set: '+best_acc)
    
    # Test on sklearn
    sk_tree = tree.DecisionTreeClassifier()
    
    sk_tree.fit(X_train, Y_train)
    
    preds = sk_tree.predict(X_test)
    
    correct = 0
    for pred, y in zip(preds, Y_test):
        if pred == y:
            correct += 1
    
    print('\n** SKLEARN **')
    print('\tAccuracy: '+str(correct/len(preds)))
    
main()