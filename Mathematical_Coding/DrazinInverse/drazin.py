# drazin.py
"""Volume 1: The Drazin Inverse.
<Mark Rose>
<Section 2>
<4/1/19>
"""

import numpy as np
from scipy import linalg as la
from scipy.sparse.csgraph import laplacian as L
import numpy.linalg as npl
#import pandas as pd


# Helper function for problems 1 and 2.
def index(A, tol=1e-5):
    """Compute the index of the matrix A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        k (int): The index of A.
    """

    # test for non-singularity
    if not np.isclose(la.det(A), 0):
        return 0
    
    n = len(A)
    k = 1
    Ak = A.copy()
    while k <= n:
        r1 = np.linalg.matrix_rank(Ak)
        r2 = np.linalg.matrix_rank(np.dot(A,Ak))
        if r1 == r2:
            return k
        Ak = np.dot(A,Ak)
        k += 1

    return k


# Problem 1
def is_drazin(A, Ad, k):
    """Verify that a matrix Ad is the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.
        Ad ((n,n) ndarray): A candidate for the Drazin inverse of A.
        k (int): The index of A.

    Returns:
        (bool) True of Ad is the Drazin inverse of A, False otherwise.
    """
    #Check to make sure all the conditions hold. Return true if they do, false if they don't
    if (np.allclose(A @ Ad, Ad @ A) and np.allclose(npl.matrix_power(A,k+1) @ Ad, npl.matrix_power(A,k)) and np.allclose(Ad @ A @ Ad, Ad)):
        return(True)
    else:
        return(False)

# Problem 2
def drazin_inverse(A, tol=1e-4):
    """Compute the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
       ((n,n) ndarray) The Drazin inverse of A.
    """
    ## Initialize f1 and f2 sorting functions.
    f1 = lambda x: np.abs(x) > tol
    f2 = lambda x: np.abs(x) <= tol

    ## Grab shape size of A, compute Q1,Q2,S,T,k1,k2 from Schur function.
    n = A.shape[0]
    Q1,S,k1 = la.schur(A,sort=f1)
    Q2,T,k2 = la.schur(A,sort=f2)

    ## Compute U, U_inv, V and Z matrices using Alg 15.1
    U = np.hstack((S[:,:k1],T[:,:(n-k1)]))
    U_inv = la.inv(U)
    V = U_inv @ A @ U
    Z = np.zeros((n,n))

    ## If singular, recompute Z matrix.
    if k1 != 0:
        M_inv = la.inv(V[:k1,:k1])
        Z[:k1,:k1] = M_inv

    ## Return solution from algorithm 15.1
    return U @ Z @ U_inv
    raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def effective_resistance(A):
    """Compute the effective resistance for each node in a graph.

    Parameters:
        A ((n,n) ndarray): The adjacency matrix of an undirected graph.

    Returns:
        ((n,n) ndarray) The matrix where the ijth entry is the effective
        resistance from node i to node j.
    """
    #Initialize a resistance matrix
    n = len(A)
    R = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            #Change the resistance for each value
            if i != j:
                lap = L(A)
                lap[j] = np.eye(n)[j]
                lap = drazin_inverse(lap)
                R[i][j] = lap[i][i]
    #return the Matrix
    return(R)
                
    raise NotImplementedError("Problem 3 Incomplete")


# Problems 4 and 5
class LinkPredictor:
    """Predict links between nodes of a network."""

    def __init__(self, filename='social_network.csv'):
        """Create the effective resistance matrix by constructing
        an adjacency matrix.
[
        Parameters:
            filename (str): The name of a file containing graph data.
        """
        #Read in the data
        data = pd.read_csv(filename,header=None)
        first_col = list(data[0])
        second_col = list(data[1])
        #Get
        all_names = first_col.copy()
        all_names.extend(second_col)
        labels = list(set(all_names))
        n = len(labels)
        name_ind = {}
        val_ind = {}
        #Create the name indices
        for i in range(n):
            name_ind[labels[i]] = i
            val_ind[i] = labels[i]
        #Create attributes to save the adjacency and spot of the name in the matrix
        self.name_ind = name_ind
        self.val_ind = val_ind
        self.adjacency = np.zeros((n,n))
        for i in range(len(first_col)):
            #Change the adjacency matrix
            val1 = name_ind[first_col[i]]
            val2 = name_ind[second_col[i]]
            self.adjacency[val1][val2] +=1
            self.adjacency[val2][val1] +=1
        #Save the names and resistance matrix
        self.names = labels
        self.resistance = effective_resistance(self.adjacency)


    def predict_link(self, node=None):
        """Predict the next link, either for the whole graph or for a
        particular node.

        Parameters:
            node (str): The name of a node in the network.

        Returns:
            node1, node2 (str): The names of the next nodes to be linked.
                Returned if node is None.
            node1 (str): The name of the next node to be linked to 'node'.
                Returned if node is not None.

        Raises:
            ValueError: If node is not in the graph.
        """
        #Raise an error if the node is not in the graph
        if node!=None and self.name_ind.get(node) == None:
            raise ValueError("Node is not in the graph")
        #Zero out entries in the adjacency with values that are not zero
        mask = (self.adjacency == 0)
        A = mask*self.resistance
        #If the node is none, find the minimum value other than the zero values in the resistance matrix
        if node == None:
            minval = np.min(A[np.nonzero(A)])
            loc = np.where(A==minval)
            print(loc[0][0])
            return(self.val_ind[loc[0][0]], self.val_ind[loc[1][0]])
        #If the node has a string, find the minimum value other than the zero values in that row of the resistance matrix
        else:
            val = self.name_ind[node]
            minval = np.min(A[val][np.nonzero(A[val])])
            loc = np.where(A==minval)
            return(self.val_ind[loc[1][0]])


    def add_link(self, node1, node2):
        """Add a link to the graph between node 1 and node 2 by updating the
        adjacency matrix and the effective resistance matrix.

        Parameters:
            node1 (str): The name of a node in the network.
            node2 (str): The name of a node in the network.

        Raises:
            ValueError: If either node1 or node2 is not in the graph.
        """
        #Don't add if the names are not in the matrix
        if self.name_ind.get(node1)==None or self.name_ind.get(node2)==None:
            raise ValueError('One of the Nodes is Not in the Graph')
        ind1 = self.name_ind[node1]
        ind2 = self.name_ind[node2]
        #Update the adjacency and resistance matrix
        self.adjacency[ind1][ind2]+=1
        self.adjacency[ind2][ind1]+=1
        self.resistance = effective_resistance(self.adjacency)
