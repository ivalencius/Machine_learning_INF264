# Creates structures for node and a tree

class Node:
    def __init__(self):
        # Condition must return true or false
        self.condition = None
        self.left = None
        self.right = None
        # Node gets label ONLY when it is a leaf
        self.label = None
        self.feature_idx = None
        self.mean = None
        # self.weight = None
        
    def clear_node(self):
        self.condition = None
        self.left = None
        self.right = None
        # Node gets label ONLY when it is a leaf
        self.label = None
        self.feature_idx = None
        self.mean = None
        
    def make_leaf(self, label):
        self.label = label
        if self.left or self.right != None:
            print("ERROR: THIS LEAF HAS CHILDREN")

    def get_label(self):
        return self.label
    
    def is_leaf(self):
        return self.label != None
    
    def has_children(self):
        if self.right == None and self.left == None:
            return False
        else:
            return True
        
    # Given a condition, add it 
    def add_condition(self, feature_idx, mean):
        self.feature_idx = feature_idx
        self.mean = mean
        
    def test_condition(self, x):
         return x[self.feature_idx] >= self.mean
     
    # Returns the label of the Node if it is a leaf or else 
    # branches based on condition and classifies via next Node
    def predict(self, x):
        if self.is_leaf():
            return chr(self.label)
        test = self.test_condition(x)
        if test:
            val = self.right.predict(x) # Condition was true, branch right
        else:
            val = self.left.predict(x)
        return val
    
    # Sets left and right to point to new nodes
    def set_children(self, nodeL, nodeR):
        self.left = nodeL
        self.right = nodeR
    
    # # Number of training points reaching the node
    # def set_weight(self, count):
    #     self.weight = count
        
    # def get_weight(self):
    #     return self.weight
    
    # Prints all children nodes
    def print_Nodes(self):
        if self.is_leaf():
             print('LEAF LABEL: '+str(self.label))
        else:
            right_child = 'R: \n'
            left_child ='L: \n'
            print('NODE: ')
            print(right_child)
            self.right.print_Nodes()
            print(left_child)
            self.left.print_Nodes()
            return right_child, left_child
            # print('split on feature: '+str(self.cond))
                  