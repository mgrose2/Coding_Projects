# binary_trees.py
"""Volume 2: Binary Trees.
<Mark Rose>
<Section 2>
<10/17/18>
"""

# These imports are used in BST.draw().
import random
import time
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


class SinglyLinkedListNode:
    """A node with a value and a reference to the next node."""
    def __init__(self, data):
        self.value, self.next = data, None

class SinglyLinkedList:
    """A singly linked list with a head and a tail."""
    def __init__(self):
        self.head, self.tail = None, None

    def append(self, data):
        """Add a node containing the data to the end of the list."""
        n = SinglyLinkedListNode(data)
        if self.head is None:
            self.head, self.tail = n, n
        else:
            self.tail.next = n
            self.tail = n

    def iterative_find(self, data):
        """Search iteratively for a node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        current = self.head
        while current is not None:
            if current.value == data:
                return current
            current = current.next
        raise ValueError(str(data) + " is not in the list")

    # Problem 1
    def recursive_find(self, data):
        """Search recursively for the node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        def my_step(current):
            if current == None:
                raise ValueError('The value could not be found.')                       #Searches recursively in the tree for the given value.
            elif current.value == data:
                return(current)
            else:
                return(my_step(current.next))
        return (my_step(self.head))                                                     #Starts with the root and goes down the correct subtree by checking the inequalities. 
        raise NotImplementedError("Problem 1 Incomplete")


class BSTNode:
    """A node class for binary search trees. Contains a value, a
    reference to the parent node, and references to two child nodes.
    """
    def __init__(self, data):
        """Construct a new node and set the value attribute. The other
        attributes will be set when the node is added to a tree.
        """
        self.value = data
        self.prev = None        # A reference to this node's parent node.
        self.left = None        # self.left.value < self.value
        self.right = None       # self.value < self.right.value


class BST:
    """Binary search tree data structure class.
    The root attribute references the first node in the tree.
    """
    def __init__(self):
        """Initialize the root attribute."""
        self.root = None

    def find(self, data):
        """Return the node containing the data. If there is no such node
        in the tree, including if the tree is empty, raise a ValueError.
        """

        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until the node containing
            the data is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree.")
            if data == current.value:               # Base case 2: data found!
                return current
            if data < current.value:                # Recursively search left.
                return _step(current.left)
            else:                                   # Recursively search right.
                return _step(current.right)

        # Start the recursion on the root of the tree.
        return _step(self.root)

    # Problem 2
    def insert(self, data):
        """Insert a new node containing the specified data.

        Raises:
            ValueError: if the data is already in the tree.

        Example:
            >>> tree = BST()                    |
            >>> for i in [4, 3, 6, 5, 7, 8, 1]: |            (4)
            ...     tree.insert(i)              |            / \
            ...                                 |          (3) (6)
            >>> print(tree)                     |          /   / \
            [4]                                 |        (1) (5) (7)
            [3, 6]                              |                  \
            [1, 5, 7]                           |                  (8)
            [8]                                 |
        """
        if self.root == None:
            new_node = BSTNode(data)                                                                #Starts with the root if it does not already have a value.
            self.root = new_node
            return
        def my_step(data, current_node):
            if current_node == None:
                current_node = BSTNode(data)
                return(current_node)
            elif current_node.value == data:
                raise ValueError('The value is already in the tree.')                               #Traverses down the tree until either there is an empty spot, or the value is already there. If so, an error is raised.
            elif data < current_node.value:
                current_node.left=my_step(data, current_node.left)
                current_node.left.prev = current_node
                return(current_node)
            elif data > current_node.value:
                current_node.right=my_step(data, current_node.right)
                current_node.right.prev = current_node
                return(current_node)
        my_step(data, self.root)
        return

        raise NotImplementedError("Problem 2 Incomplete")

    # Problem 3
    def remove(self, data):
        """Remove the node containing the specified data.

        Raises:
            ValueError: if there is no node containing the data, including if
                the tree is empty.

        Examples:
            >>> print(12)                       | >>> print(t3)
            [6]                                 | [5]
            [4, 8]                              | [3, 6]
            [1, 5, 7, 10]                       | [1, 4, 7]
            [3, 9]                              | [8]
            >>> for x in [7, 10, 1, 4, 3]:      | >>> for x in [8, 6, 3, 5]:
            ...     t1.remove(x)                | ...     t3.remove(x)
            ...                                 | ...
            >>> print(t1)                       | >>> print(t3)
            [6]                                 | [4]
            [5, 8]                              | [1, 7]
            [9]                                 |
                                                | >>> print(t4)
            >>> print(t2)                       | [5]
            [2]                                 | >>> t4.remove(1)
            [1, 3]                              | ValueError: <message>
            >>> for x in [2, 1, 3]:             | >>> t4.remove(5)
            ...     t2.remove(x)                | >>> print(t4)
            ...                                 | []
            >>> print(t2)                       | >>> t4.remove(5)
            []                                  | ValueError: <message>
        """
        my_node = self.find(data)                                                   #Searches for an item in the tree and will raise an error if not found. If found, my_node becomes that node.
        def remove_from_tree(my_node):
            if my_node.left == None and my_node.right == None:
                if my_node == self.root:
                    self.root = None                                                #Checks for root base case, and then moves by changing the predecessors values if there are no children.
                    return
                elif my_node == my_node.prev.left:
                    my_node.prev.left = None
                    return
                else:
                    my_node.prev.right = None
                    return
            elif my_node.left != None and my_node.right != None:                    #If there are children on the left and the right of the node, we find a replacement by going to the furthest right child of the left subtree.
                pred = my_node.left
                while(pred.right != None):
                    pred = pred.right
                my_node.value = pred.value
                remove_from_tree(pred)
                return
            else:
                if my_node == self.root:                                            
                    if my_node.left != None:
                        my_node.left.prev = None
                        self.root = my_node.left
                    else:
                        my_node.right.prev = None
                        self.root = my_node.right
                elif my_node.prev.left == my_node:                                   #Checks first for the root basecase, and then if not changes the prececessors and children's values.
                    if my_node.left != None:
                        my_node.prev.left = my_node.left
                        my_node.left.prev = my_node.prev
                    else:
                        my_node.prev.left = my_node.right
                        my_node.right.prev = my_node.prev
                else:
                    if my_node.left != None:
                        my_node.prev.right = my_node.left
                        my_node.left.prev = my_node.prev
                    else:
                        my_node.prev.right = my_node.right
                        my_node.right.prev = my_node.prev
        remove_from_tree(my_node)                                                          
        return
        raise NotImplementedError("Problem 3 Incomplete")

    def __str__(self):
        """String representation: a hierarchical view of the BST.

        Example:  (3)
                  / \     '[3]          The nodes of the BST are printed
                (2) (5)    [2, 5]       by depth levels. Edges and empty
                /   / \    [1, 4, 6]'   nodes are not printed.
              (1) (4) (6)
        """
        if self.root is None:                       # Empty tree
            return "[]"
        out, current_level = [], [self.root]        # Nonempty tree
        while current_level:
            next_level, values = [], []
            for node in current_level:
                values.append(node.value)
                for child in [node.left, node.right]:
                    if child is not None:
                        next_level.append(child)
            out.append(values)
            current_level = next_level
        return "\n".join([str(x) for x in out])

    def draw(self):
        """Use NetworkX and Matplotlib to visualize the tree."""
        if self.root is None:
            return

        # Build the directed graph.
        G = nx.DiGraph()
        G.add_node(self.root.value)
        nodes = [self.root]
        while nodes:
            current = nodes.pop(0)
            for child in [current.left, current.right]:
                if child is not None:
                    G.add_edge(current.value, child.value)
                    nodes.append(child)

        # Plot the graph. This requires graphviz_layout (pygraphviz).
        nx.draw(G, pos=graphviz_layout(G, prog="dot"), arrows=True,
                with_labels=True, node_color="C1", font_size=8)
        plt.show()


class AVL(BST):
    """Adelson-Velsky Landis binary search tree data structure class.
    Rebalances after insertion when needed.
    """
    def insert(self, data):
        """Insert a node containing the data into the tree, then rebalance."""
        BST.insert(self, data)      # Insert the data like usual.
        n = self.find(data)
        while n:                    # Rebalance from the bottom up.
            n = self._rebalance(n).prev

    def remove(*args, **kwargs):
        """Disable remove() to keep the tree in balance."""
        raise NotImplementedError("remove() is disabled for this class")

    def _rebalance(self,n):
        """Rebalance the subtree starting at the specified node."""
        balance = AVL._balance_factor(n)
        if balance == -2:                                   # Left heavy
            if AVL._height(n.left.left) > AVL._height(n.left.right):
                n = self._rotate_left_left(n)                   # Left Left
            else:
                n = self._rotate_left_right(n)                  # Left Right
        elif balance == 2:                                  # Right heavy
            if AVL._height(n.right.right) > AVL._height(n.right.left):
                n = self._rotate_right_right(n)                 # Right Right
            else:
                n = self._rotate_right_left(n)                  # Right Left
        return n

    @staticmethod
    def _height(current):
        """Calculate the height of a given node by descending recursively until
        there are no further child nodes. Return the number of children in the
        longest chain down.
                                    node | height
        Example:  (c)                  a | 0
                  / \                  b | 1
                (b) (f)                c | 3
                /   / \                d | 1
              (a) (d) (g)              e | 0
                    \                  f | 2
                    (e)                g | 0
        """
        if current is None:     # Base case: the end of a branch.
            return -1           # Otherwise, descend down both branches.
        return 1 + max(AVL._height(current.right), AVL._height(current.left))

    @staticmethod
    def _balance_factor(n):
        return AVL._height(n.right) - AVL._height(n.left)

    def _rotate_left_left(self, n):
        temp = n.left
        n.left = temp.right
        if temp.right:
            temp.right.prev = n
        temp.right = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_right_right(self, n):
        temp = n.right
        n.right = temp.left
        if temp.left:
            temp.left.prev = n
        temp.left = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_left_right(self, n):
        temp1 = n.left
        temp2 = temp1.right
        temp1.right = temp2.left
        if temp2.left:
            temp2.left.prev = temp1
        temp2.prev = n
        temp2.left = temp1
        temp1.prev = temp2
        n.left = temp2
        return self._rotate_left_left(n)

    def _rotate_right_left(self, n):
        temp1 = n.right
        temp2 = temp1.left
        temp1.left = temp2.right
        if temp2.right:
            temp2.right.prev = temp1
        temp2.prev = n
        temp2.right = temp1
        temp1.prev = temp2
        n.right = temp2
        return self._rotate_right_right(n)


# Problem 4
def prob4():
    """Compare the build and search times of the SinglyLinkedList, BST, and
    AVL classes. For search times, use SinglyLinkedList.iterative_find(),
    BST.find(), and AVL.find() to search for 5 random elements in each
    structure. Plot the number of elements in the structure versus the build
    and search times. Use log scales where appropriate.
    """
    num_list = [2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10]
    
    build_list = []
    build_binary = []
    build_avl = []
    
    find_list = []
    find_binary = []
    find_avl = []
    
    with open('english.txt','r') as myfile:
        mylist = myfile.readlines()
    for n in num_list:
        a = random.sample(mylist,n)            #Random sample of n elements
        my_single = SinglyLinkedList()                                                          #Time each build starting with list
        my_binary = BST()
        my_avl = AVL()
        start_list = time.time()
        for i in a:
            my_single.append(i)
        list_end = time.time()-start_list
        build_list.append(list_end)
        
        start_binary = time.time()
        for i in a:
            my_binary.insert(i)
        binary_end = time.time()-start_binary
        build_binary.append(binary_end)
        
        start_avl = time.time()
        for i in a:
            my_avl.insert(i)
        avl_end = time.time()-start_avl                                                         #Time each find function startin with iterative_find
        build_avl.append(avl_end)
        
        b = random.sample(a,5)                #Random sample of 5 elements
        start_list = time.time()
        for i in b:
            my_single.iterative_find(i)
        list_end = time.time()-start_list
        find_list.append(list_end)
        
        start_binary = time.time()
        for i in b:
            my_binary.find(i)
        binary_end = time.time()-start_binary
        find_binary.append(binary_end)
        
        start_avl = time.time()
        for i in b:
            my_avl.find(i)
        avl_end = time.time()-start_avl
        find_avl.append(avl_end)
    
    fig, axes= plt.subplots(1,2)
    axes[0].loglog(num_list, build_list, 'b-o', label = 'List Build Times',basex=2, basey=10)
    axes[0].loglog(num_list, build_binary, 'g-o', label = 'BST Build Times',basex=2, basey=10)                          #Graph all of the build times on one axes
    axes[0].loglog(num_list, build_avl, 'r-o', label = 'AVL Build Times',basex=2, basey=10)
    axes[0].legend()
    axes[0].set_title('Build Execution Times')
    axes[0].set_xlabel('n')
    axes[0].set_ylabel('Seconds')
    axes[1].loglog(num_list, find_list, 'b-o', label = 'List Find Times',basex=2, basey=10)
    axes[1].loglog(num_list, find_binary, 'g-o', label = 'BST Find Times',basex=2, basey=10)                      #Graph all of the find times on another axes
    axes[1].loglog(num_list, find_avl, 'r-o', label = 'AVL Find Times',basex=2, basey=10)
    axes[1].legend()
    axes[1].set_title('Find Execution Times')
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('Seconds')
    
    plt.show()
    return
    raise NotImplementedError("Problem 4 Incomplete")
