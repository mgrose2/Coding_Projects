# nearest_neighbor.py
"""Volume 2: Nearest Neighbor Search.
<Mark Rose>
<Section 2>
<10/24/18>
"""


import numpy as np
from scipy import linalg as la
from scipy.spatial import KDTree
from scipy import stats
from matplotlib import pyplot as plt

# Problem 1
def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    dist = la.norm(X-z, axis=1)                                                                          #Broadcast the rows against the input vector
    return(X[np.argmin(dist)], min(dist))                           
    raise NotImplementedError("Problem 1 Incomplete")


# Problem 2: Write a KDTNode class.
class KDTNode:
    def __init__(self, x):
        if type(x) != np.ndarray:
            raise TypeError("x is not the correct type")
        self.value = x
        self.left = None
        self.right = None                                                                               #Creates a KDTNode and saves the attributes
        self.pivot = None
        self.prev = None

# Problems 3 and 4
class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    # Problem 3
    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
        """
        if type(data) != np.ndarray or np.shape(data) != (len(data),):
            raise ValueError("The data type is incorrect")
        
        if self.root == None:
            new_node = KDTNode(data)                                                                #Sets the root if it is empty.
            new_node.pivot = 0
            self.root = new_node
            self.k = len(data)
            return
        def my_step(data, current_node, parent_level):
            if current_node == None:
                current_node = KDTNode(data)
                if (parent_level) == len(data) -1:
                    current_node.pivot = 0
                else:
                    current_node.pivot = parent_level+1
                return(current_node)
            elif current_node.value[current_node.pivot] == data[current_node.pivot]:
                raise ValueError('The value is already in the tree.')                               #Traverses down the tree until either there is an empty spot, or the value is already there. If so, an error is raised.
            elif data[current_node.pivot] < current_node.value[current_node.pivot]:
                current_node.left=my_step(data, current_node.left, current_node.pivot)
                current_node.left.prev = current_node
                return(current_node)
            elif data[current_node.pivot] > current_node.value[current_node.pivot]:
                current_node.right=my_step(data, current_node.right, current_node.pivot)
                current_node.right.prev = current_node
                return(current_node)
        my_step(data, self.root, self.root.pivot)
        return
            
        raise NotImplementedError("Problem 3 Incomplete")

    # Problem 4
    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        root = self.root
        def KDSearch(current, nearest, dp):
            if current == None:
                return(nearest, dp)
            x = current.value
            i = current.pivot
            if la.norm(x-z) < dp:
                nearest = current
                dp = la.norm(x-z)
            if z[i] < x[i]:                                                                                 
                nearest, dp = KDSearch(current.left, nearest, dp)
                if z[i] + dp >= x[i]:
                    nearest, dp = KDSearch(current.right, nearest, dp)                                     #checks the sphere of radius to see if it should check the right subtree
            else:
                nearest, dp = KDSearch(current.right, nearest, dp)
                if z[i] - dp <= x[i]:                                                                       #checks the sphere of radius to see if it should check the left subtree
                    nearest, dp = KDSearch(current.left, nearest, dp)                                       
            return(nearest, dp)
        node, dp = KDSearch(root, root, la.norm(root.value-z))
        return (node.value, dp)

    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


# Problem 5: Write a KNeighborsClassifier class.
class KNeighborsClassifier:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors                                                      #initializes the amount of neighbors we would like to find
        
    def fit(self, X, y):
        self.tree = KDTree(X)
        self.labels = y                                                                     #stores a tree of the x_train and the labels of the y_train
        return
    
    def predict(self, z):
        distances, indices = self.tree.query(z, k=self.n_neighbors)
        if(self.n_neighbors ==1):
            return(self.labels[indices])
        winner = stats.mode(self.labels[indices], axis=1)[0][:,0]                           #takes the mode of k values and returns those values
        return(winner)
    
    
            


# Problem 6
def prob6(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    data = np.load(filename)
    X_train = data['X_train'].astype(np.float)
    y_train = data['y_train']
    X_test = data['X_test'].astype(np.float)
    y_test = data['y_test']
    
    model = KNeighborsClassifier(n_neighbors)
    model.fit(X_train, y_train)
    my_pred = model.predict(X_test)
    accuracy = (my_pred) == y_test                                                          #check to see if my predictions are equal to the y_truth values. a matrix of 1's (correct) and 0's (incorrect) is made
    return(np.mean(accuracy)*100)                                                           #takes the average of the matrix to calculate an accuracy
    
    raise NotImplementedError("Problem 6 Incomplete")
