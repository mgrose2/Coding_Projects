# solutions.py
"""Volume 1: The Page Rank Algorithm.
<Mark Rose>
<Section 2>
<3/4/19>
"""
import numpy as np
import pandas as pd
import scipy.linalg as la
import networkx as nx
from itertools import combinations

# Problems 1-2
class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        (fill this out after completing DiGraph.__init__().)
    """
    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        #Make sure there are the corerct number of labels
        n = len(A)
        if labels != None and len(labels)!=n:
            raise ValueError("You have the wrong number of labels")
        #Check for any holes and fill them if there are
        for i in range(n):
            if np.all(A[:,i]==0):
                A[:,i] = np.ones_like(A[:,i])
        #Normalize the matrix A
        A = A/np.sum(A,axis=0)
        if labels == None:
            labels = []
            for i in range(n):
                labels.append(str(i))
        #Set everything as attributes
        self.labels = labels
        self.A = A
        return
        
    # Problem 2
    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        #Use the equation given in the book
        p = la.inv(np.eye(len(self.A))-epsilon*self.A) @ ((1-epsilon)/len(self.A) * np.ones(len(self.A)))
        my_dic = {}
        #Create a dictionary of the vales and the page rank
        for i,val in enumerate(self.labels):
            my_dic[val] = p[i]
        return(my_dic)
        raise NotImplementedError("Problem 2 Incomplete")

    # Problem 2
    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        #Get the eigenvectors of the A matrix
        n = len(self.A)
        vals, vecs = la.eig(epsilon*self.A+((1-epsilon)/n*np.ones((n,n))))
        p = vecs[:,np.argmax(vals)]
        #Normalize P
        p = p/np.sum(p)
        my_dic = {}
        #Create the dictionary for the page ranks values
        for i,val in enumerate(self.labels):
            my_dic[val] = p[i]
        return(my_dic)
        raise NotImplementedError("Problem 2 Incomplete")

    # Problem 2
    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        n = len(self.A)
        p = np.ones(n)/n
        #Iterate to find the new p value
        for i in range(maxiter):
            p_new = epsilon*self.A @ p + (1-epsilon)/n*np.ones(n)
            #Break if the norm is less than the tolerance
            if la.norm(p_new-p) < tol:
                break
            p = p_new
        #Create the dictionary for the page rank values
        my_dic = {}
        for i,val in enumerate(self.labels):
            my_dic[val] = p_new[i]
        return(my_dic)
        raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """
    #Sort the page rank values in the dictionary
    my_list = []
    past_list=sorted(d.items(), key=lambda x: x[1], reverse=True)
    for i in past_list:
        #Append the sorted labels into a list
        my_list.append(str(i[0]))
    return(my_list)
    raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def rank_websites(filename="web_stanford.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks().

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.
    """
    #Get all of the text and put it into a string
    info = open(filename, 'r')
    my_info = info.read()
    #Take out all of the dashes and end of line characters
    my_info = my_info.replace('\n', '/')
    my_info=my_info.split('/')
    my_info.remove('')
    #Create a set of the words and sort them
    my_info = set(my_info)
    my_info = sorted(my_info)
    info = open(filename, 'r')
    info = info.readlines()
    #Initialize the labels and the A matrix
    labels = []
    labels_ind = {}
    n = len(my_info)
    A = np.zeros((n,n))
    #Take out all of the dashes and end of line characters
    for i in range(len(info)):
        info[i]=info[i].split('/')
        info[i][-1] = info[i][-1][0:-1]
    #Create the labels and the dictionary of the indices
    for i in range(n):
        labels.append(my_info[i])
        labels_ind[my_info[i]] = i
    #Add the weight of 1 to each of the connections in the Matrix
    for i in range(len(info)):
        conn = [labels_ind[k] for k in info[i][1:]]
        j= labels_ind[info[i][0]]
        A[j,conn] = 1
    #Use the previous designed  functions to get the page rank and return it
    my_page = DiGraph(A.T,labels)
    rank = my_page.itersolve(epsilon)
    ranked_list = get_ranks(rank)
    return(ranked_list)
    
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def rank_ncaa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.
    """
    #Read in the data usig a panda dataframe
    data = pd.read_csv(filename)
    winner = list(data['Winner'])
    loser = list(data['Loser'])
    #Get all of the unique teams by combining the winners and losers into a set and sort it
    all_teams = winner.copy()
    all_teams.extend(loser)
    labels = set(all_teams)
    labels = sorted(labels)
    #Initialize A and the team_indices
    n = len(labels)
    A = np.zeros((n,n))
    team_ind = {}
    #Create the team indices
    for i in range(n):
        team_ind[labels[i]] = i
    #Add weights to the directed graphs
    for i in range(len(winner)):
        win = team_ind[winner[i]]
        lose = team_ind[loser[i]]
        A[win,lose]+=1
    #Use the predefined functions to get the answers and return
    my_page = DiGraph(A,labels)
    rank = my_page.itersolve(epsilon)
    ranked_list = get_ranks(rank)
    return(ranked_list)
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def rank_actors(filename="top250movies.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node a points to
    node b with weight w if actor a and actor b were in w movies together but
    actor b was listed first. Use NetworkX to compute the PageRank values of
    the actors, then rank them with get_ranks().

    Each line of the file has the format
        title/actor1/actor2/actor3/...
    meaning actor2 and actor3 should each have an edge pointing to actor1,
    and actor3 should have an edge pointing to actor2.
    """
    #Initialize a DiGraph using the predefined module and read in the data
    DG = nx.DiGraph()
    info = open(filename, 'r', encoding='utf-8')
    info = open(filename, 'r')
    info = info.read()
    info = info.splitlines()
    #Split all of the names and get all of the combinations, without the title of the movie
    for i in range(len(info)):
        info[i] = info[i].split('/')
        my_list = list(combinations(info[i][1:],2))
    #Add weight to each of the edges depending on the order of the actor
        for j in my_list:
            if DG.has_edge(j[1],j[0]) == True:
                DG[j[1]][j[0]]['weight']+=1
            else:
                DG.add_edge(j[1], j[0], weight=1)
    #Get the page rank using the module's method
    p = nx.pagerank(DG, alpha = epsilon)
    return(get_ranks(p))
    raise NotImplementedError("Problem 6 Incomplete")
