# breadth_first_search.py
"""Volume 2: Breadth-First Search.
<Mark Rose>
<Section 2>
<10/31/18>
"""


# Problems 1-3
class Graph:
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a set of
    the corresponding node's neighbors.

    Attributes:
        d (dict): the adjacency dictionary of the graph.
    """
    def __init__(self, adjacency={}):
        """Store the adjacency dictionary as a class attribute"""
        self.d = dict(adjacency)

    def __str__(self):
        """String representation: a view of the adjacency dictionary."""
        return str(self.d)

    # Problem 1
    def add_node(self, n):
        """Add n to the graph (with no initial edges) if it is not already
        present.

        Parameters:
            n: the label for the new node.
        """
        if n not in keys:
            self.d.update({n: set()})                                                           #add to the dictionary if empty
        return
        raise NotImplementedError("Problem 1 Incomplete")

    # Problem 1
    def add_edge(self, u, v):
        """Add an edge between node u and node v. Also add u and v to the graph
        if they are not already present.

        Parameters:
            u: a node label.
            v: a node label.
        """
        if u not in self.d.keys():
            add_node(u)
        if v not in self.d.keys():
            add_node(v)                                                                       #check to see if the nodes are in the dictionary and if not add them
        if v not in self.d[u]:
            values = self.d[u]
            values.add(v)
            self.d.update({u: values})
                                                                                            # Add edge from v to u if it doesn't already exist
        if u not in self.d[v]:
            values = self.d[v]
            values.add(u)
            self.d.update({v: values})
        return
        raise NotImplementedError("Problem 1 Incomplete")

    # Problem 1
    def remove_node(self, n):
        """Remove n from the graph, including all edges adjacent to it.

        Parameters:
            n: the label for the node to remove.

        Raises:
            KeyError: if n is not in the graph.
        """
        if n not in self.d.keys():
            raise KeyError("Node is not in graph.")
                                                                            #if in the dictionary, pop off n value and all its edges.
        self.d.pop(n)
        for j in list(self.d.values()):
            j.discard(n)
        return
        
        raise NotImplementedError("Problem 1 Incomplete")

    # Problem 1
    def remove_edge(self, u, v):
        """Remove the edge between nodes u and v.

        Parameters:
            u: a node label.
            v: a node label.

        Raises:
            KeyError: if u or v are not in the graph, or if there is no
                edge between u and v.
        """
        if u not in self.d.keys() or v not in self.d.keys():
            raise KeyError("Edge does not exist in graph.")
                                                                            # Remove edge v to u, otherwise return KeyError
        if u in self.d[v]:
            self.d[v].remove(u)
        else:
            raise KeyError("There is no edge between u and v.")
                                                                            # Remove edge u to v, otherwise return KeyError
        if v in self.d[u]:
            self.d[u].remove(v)
        else:
            raise KeyError("There is no edge between u and v.")
        return
        raise NotImplementedError("Problem 1 Incomplete")

    # Problem 2
    def traverse(self, source):
        """Traverse the graph with a breadth-first search until all nodes
        have been visited. Return the list of nodes in the order that they
        were visited.

        Parameters:
            source: the node to start the search at.

        Returns:
            (list): the nodes in order of visitation.

        Raises:
            KeyError: if the source node is not in the graph.
        """
        if source not in self.d.keys():				#source node not in the graph
            raise KeyError("Node not found in the graph.")
        
        V = []
        Q = [source]
        M = set(source)
        
        while Q:									#pop node off Q
            node = Q.pop(0)
            if node not in V:						#add node to B
                V.append(node)
                neighbors = self.d[node]			#find neighbors of node
                for n in neighbors:
                    if n not in M:					#add neighbors to M and Q
                        M.add(n)
                        Q.append(n)                    
        return V
        raise NotImplementedError("Problem 2 Incomplete")

    # Problem 3
    def shortest_path(self, source, target):
        """Begin a BFS at the source node and proceed until the target is
        found. Return a list containing the nodes in the shortest path from
        the source to the target, including endoints.

        Parameters:
            source: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from source to target,
                including the endpoints.

        Raises:
            KeyError: if the source or target nodes are not in the graph.
        """
        if source not in self.d.keys():				#source node not in the graph
            raise KeyError("Node not found in the graph.")
        if target not in self.d.keys():				#target node not in the graph
            raise KeyError("Node not found in the graph.")
        V = set()
        Q = [source]

        ## Run while loop as long as values exist on queue. 
        while Q:
            path = Q.pop(0)
            node = path[-1]
            ## Return total path if node is found.
            if node == target:
                return path
            ## If node is not found, do a BFS in neighbors of node.
            elif node not in V:
                for adjacent in self.d.get(node, []):
                    ## Append new neighbors to the certain path then add their 
                    ## paths to Q.
                    new_path = list(path)
                    new_path.append(adjacent)
                    Q.append(new_path)
                V.add(node)
                         
        return V    
        raise NotImplementedError("Problem 3 Incomplete")


# Problems 4-6
class MovieGraph:
    """Class for solving the Kevin Bacon problem with movie data from IMDb."""

    # Problem 4
    def __init__(self, filename="movie_data.txt"):
        """Initialize a set for movie titles, a set for actor names, and an
        empty NetworkX Graph, and store them as attributes. Read the speficied
        file line by line, adding the title to the set of movies and the cast
        members to the set of actors. Add an edge to the graph between the
        movie and each cast member.

        Each line of the file represents one movie: the title is listed first,
        then the cast members, with entries separated by a '/' character.
        For example, the line for 'The Dark Knight (2008)' starts with

        The Dark Knight (2008)/Christian Bale/Heath Ledger/Aaron Eckhart/...

        Any '/' characters in movie titles have been replaced with the
        vertical pipe character | (for example, Frost|Nixon (2008)).
        """
        self.bacon = nx.Graph()					#Initialize the NetworkX Graph
        movies = set()
        actors = set()
        self.movies = movies
        self.actors = actors
        
        
        with open(filename, 'r', encoding=encoding) as my_file:		#read in data
            for line in my_file:
                splitter = line.strip().split('/')					#split lines
                self.movies.add(splitter[0])						#add movie titles
                for x in range(1, len(splitter)):
                    self.actors.add(splitter[x])					#add actors
                    self.bacon.add_edge(splitter[0], splitter[x])	#add edges
        return
        raise NotImplementedError("Problem 4 Incomplete")

    # Problem 5
    def path_to_actor(self, source, target):
        """Compute the shortest path from source to target and the degrees of
        separation between source and target.

        Returns:
            (list): a shortest path from source to target, including endpoints.
            (int): the number of steps from source to target, excluding movies.
        """
        shortest_path = nx.shortest_path(self.bacon, source, target)	#Find shortest path
        num_steps = (len(shortest_path)-1)//2							#Calculate number of steps
        
        return shortest_path, num_steps
        raise NotImplementedError("Problem 5 Incomplete")

    # Problem 6
    def average_number(self, target):
        """Calculate the shortest path lengths of every actor to the target
        (not including movies). Plot the distribution of path lengths and
        return the average path length.

        Returns:
            (float): the average path length from actor to target.
        """
        D = nx.shortest_path_length(self.bacon, target)					#Find the shortest paths
        shortest_paths = [D[x] // 2 for x in self.actors]				#Remove movies from list
        average_path = sum(shortest_paths) / len(self.actors)			#Find the average
        
        plt.hist(shortest_paths, bins=[i-.5 for i in range(8)])			#create a histogram
        plt.xlabel("Length of Path")
        plt.ylabel("Number of Actors")
        plt.title(str(target))
        plt.show()
        
        return average_path
        raise NotImplementedError("Problem 6 Incomplete")
