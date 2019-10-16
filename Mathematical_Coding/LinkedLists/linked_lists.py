# linked_lists.py
"""Volume 2: Linked Lists.
<Mark Rose>
<Math 320>
<9/6/18>
"""


# Problem 1
class Node:
    """A basic node class for storing data."""
    def __init__(self, data):
        """Store the data in the value attribute."""
        if type(data) != int and type(data) != str and type(data) != float:             # Only let in data that is of type string, float, or integer
            raise TypeError('The type of data must be a string, float, or integer.')
        else:
            self.value = data
  


class LinkedListNode(Node):
    """A node class for doubly linked lists. Inherits from the Node class.
    Contains references to the next and previous nodes in the linked list.
    """
    def __init__(self, data):
        """Store the data in the value attribute and initialize
        attributes for the next and previous nodes in the list.
        """
        Node.__init__(self, data)       # Use inheritance to set self.value.
        self.next = None                # Reference to the next node.
        self.prev = None                # Reference to the previous node.


# Problems 2-5
class LinkedList:
    """Doubly linked list data structure class.

    Attributes:
        head (LinkedListNode): the first node in the list.
        tail (LinkedListNode): the last node in the list.
    """
    def __init__(self):
        """Initialize the head and tail attributes by setting
        them to None, since the list is empty initially.
        """
        self.head = None                                            
        self.tail = None
        self.size = 0

    def append(self, data):
        """Append a new node containing the data to the end of the list."""
        # Create a new node to store the input data.
        new_node = LinkedListNode(data)
        if self.head is None:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node
            self.size+=1
        else:
            # If the list is not empty, place new_node after the tail.
            self.tail.next = new_node               # tail --> new_node
            new_node.prev = self.tail               # tail <-- new_node
            # Now the last node in the list is new_node, so reassign the tail.
            self.tail = new_node
            self.size+=1

    # Problem 2
    def find(self, data):
        """Return the first node in the list containing the data.

        Raises:
            ValueError: if the list does not contain the data.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.find('b')
            >>> node.value
            'b'
            >>> l.find('f')
            ValueError: <message>
        """
        current_node = self.head
        if (current_node == None):
            raise ValueError('Value not found in the linked list.')
        while(current_node != None):
            if current_node.value == data:
                return(current_node)
            else:                                                           #Go to the next node until we find the node, and return
                current_node = current_node.next
        raise ValueError('Value not found in the linked list.')             #Raise a value error if the node is not found in the list
        
        raise NotImplementedError("Problem 2 Incomplete")

    # Problem 2
    def get(self, i):
        """Return the i-th node in the list.

        Raises:
            IndexError: if i is negative or greater than or equal to the
                current number of nodes.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.get(3)
            >>> node.value
            'd'
            >>> l.get(5)
            IndexError: <message>
        """
        current_node = self.head
        if i < 0 or i >= self.size:
            raise IndexError('The Index given is out of bounds.')
        else:
            for i in range(i):
                current_node = current_node.next                            #Go to the next noe until we finish the loop and return that node
            return(current_node)
        raise NotImplementedError("Problem 2 Incomplete")

    # Problem 3
    def __len__(self):
        """Return the number of nodes in the list.

        Examples:
            >>> l = LinkedList()
            >>> for i in (1, 3, 5):
            ...     l.append(i)
            ...
            >>> len(l)
            3
            >>> l.append(7)
            >>> len(l)
            4
        """
        return(self.size)                                                   #Return size which we defined in the initializer
        raise NotImplementedError("Problem 3 Incomplete")

    # Problem 3
    def __str__(self):
        """String representation: the same as a standard Python list.

        Examples:
            >>> l1 = LinkedList()       |   >>> l2 = LinkedList()
            >>> for i in [1,3,5]:       |   >>> for i in ['a','b',"c"]:
            ...     l1.append(i)        |   ...     l2.append(i)
            ...                         |   ...
            >>> print(l1)               |   >>> print(l2)
            [1, 3, 5]                   |   ['a', 'b', 'c']
        """
        current_node = self.head
        my_string = "["
        while(current_node != None):                                        #Add brackets to the beginning and end of the list, and add commas after each data member.
            my_string+= repr(current_node.value)
            if (current_node.next != None):
                my_string+= ', '
            current_node = current_node.next
        my_string+= ']'
        return(my_string)
        raise NotImplementedError("Problem 3 Incomplete")

    # Problem 4
    def remove(self, data):
        """Remove the first node in the list containing the data.

        Raises:
            ValueError: if the list is empty or does not contain the data.

        Examples:
            >>> print(l1)               |   >>> print(l2)
            ['a', 'e', 'i', 'o', 'u']   |   [2, 4, 6, 8]
            >>> l1.remove('i')          |   >>> l2.remove(10)
            >>> l1.remove('a')          |   ValueError: <message>
            >>> l1.remove('u')          |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.remove(10)
            ['e', 'o']                  |   ValueError: <message>
        """
        my_node = self.find(data)                                           #Use our previously made find function to find the data. If the data is not in the list, the find funciton will raise an error.
        if my_node == self.head:
            self.head = self.head.next
            if self.size == 1:
                self.tail = self.head                                         
            self.size-=1
            return(data)
        elif my_node == self.tail:                                          #Check for the end cases if the node is at the beginning or the end of the list and reassign the tail or head values if so.
            self.tail = my_node.prev
            self.tail.next = None
            self.size-=1
            return(data)
        
        else:
            my_node.prev.next = my_node.next                                #Assign the previous nodes next and the next nodes previous to change the pointers in the list.
            my_node.next.prev = my_node.prev
            self.size-=1
            return(data)
        raise NotImplementedError("Problem 4 Incomplete")

    # Problem 5
    def insert(self, index, data):
        """Insert a node containing data into the list immediately before the
        node at the index-th location.

        Raises:
            IndexError: if index is negative or strictly greater than the
                current number of nodes.

        Examples:
            >>> print(l1)               |   >>> len(l2)
            ['b']                       |   5
            >>> l1.insert(0, 'a')       |   >>> l2.insert(6, 'z')
            >>> print(l1)               |   IndexError: <message>
            ['a', 'b']                  |
            >>> l1.insert(2, 'd')       |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.insert(1, 'a')
            ['a', 'b', 'd']             |   IndexError: <message>
            >>> l1.insert(2, 'c')       |
            >>> print(l1)               |
            ['a', 'b', 'c', 'd']        |
        """
        if index < 0 or index > self.size:
            raise IndexError('The index is out of bounds.')
        elif index == self.size:
            self.append(data)                                               #Check for end cases and reassign accordingly
        elif index == 0:
            new_node = LinkedListNode(data)
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
            self.size+=1
        else:                                                               #Reassign the previous nodes next and the next nodes previous to this new node. 
            my_node = self.get(index)
            new_node = LinkedListNode(data)
            new_node.next = my_node
            new_node.prev = my_node.prev
            my_node.prev.next = new_node
            my_node.prev = new_node
            self.size+=1
        return
        raise NotImplementedError("Problem 5 Incomplete")


# Problem 6: Deque class.
class Deque(LinkedList):
    def __init__(self):
        LinkedList.__init__(self)

    def pop(self):
        return(LinkedList.remove(self, self.tail.value))                    #Use the linked list functions to make the functions within the deque class. 
    
    def popleft(self):
        LinkedList.remove(self,self.head.value)
    
    def appendleft(self, data):
        LinkedList.insert(self,0, data)
    
    def remove(*args, **kwargs):
        raise NotImplementedError("Use pop() or popleft() for removal")     #Check to make sure that they use the correct commands to remove or insert. 

    def insert(*args, **kwargs):
        raise NotImplementedError("Use append() or appendleft() for insertion")

# Problem 7
def prob7(infile, outfile):
    
    """Reverse the contents of a file by line and write the results to
    another file.

    Parameters:
        infile (str): the file to read from.
        outfile (str): the file to write to.
    """
    my_stack = Deque()
    with open(infile, 'r') as my_file:                                      #Using a deque, we will take in all of the lines of a file an append them to the deque. 
        for line in my_file:
            my_stack.append(line)

    with open(outfile, 'w') as new_file:
        while my_stack.size >0:
            new_file.write(my_stack.pop())                                  #We will then pop off the each of the last lines and write them into the new file.
    return
    
    raise NotImplementedError("Problem 7 Incomplete")

