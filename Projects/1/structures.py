# Creates structures for node and a tree

class Node:
    def __init__(self):
        # Condition must return true or false
        self.condition = None
        self.left = None
        self.right = None
        # Node gets label ONLY when it is a leaf
        self.label = None
        
    def make_leaf(self, label):
        self.label = label
        if self.left or self.right != None:
            print("ERROR: THIS LEAF HAS CHILDREN")
            
    # Given a condition, add it 
    def add_condition(self, case):
        self.condition = case
        
    # Returns the label of the Node if it is a leaf or else 
    # branches based on condition and classifies via next Node
    def predict(self, x):
        if self.label != None:
            return self.label
        test = self.condition(x)
        if test:
            val = self.right.predict(x) # Condition was true, branch right
        else:
            val = self.left.predict(x)
        return val
    
    # Sets left and right to point to new nodes
    def set_children(self, nodeL, nodeR):
        self.left = nodeL
        self.right = nodeR
    
    # Prints all children nodes
    # def print_Nodes(self):
    #     if self.left or self.right == None:
    #         print('LEAF LABEL: '+str(self.label))
    #     else:
    #         print('NODE:')
                  