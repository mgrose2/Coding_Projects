# object_oriented.py
"""Python Essentials: Object Oriented Programming.
<Mark Rose>
<Math 320>
<9/19/18>
"""
import math

class Backpack:
    """A Backpack object class. Has a name, color, max size, and a list of contents.
    Attributes:
        name (str): the name of the backpack's owner.
        color (str): the color of the backpack
        max_size (int): the max size of the amount of items in the backpack 
        contents (list): the contents of the backpack.
    """

    # Problem 1: Modify __init__() and put(), and write dump().
    def __init__(self, name, color, max_size=5): 
        """Set the name, color, size, and initialize an empty list of contents.

        Parameters:
            name (str): the name of the backpack's owner.
            color (str): the color of the backpack
            max_size (int): the max amount of items that can fit in the backpack
            contents (list): the contents of the backpack
        """          
        self.name = name  
        self.color = color
        self.max_size = max_size              
        self.contents = []

    def put(self, item):
        """Add an item to the backpack's list of contents if it is not over the max size.
        """
        if len(self.contents) < self.max_size:
            self.contents.append(item) 
        else: 
            print('No Room!')

    def take(self, item):
        """Remove an item from the backpack's list of contents."""
        self.contents.remove(item)
        
    def dump(self):
        """Clear the backpack's list of contents"""
        self.contents.clear()


    # Magic Methods -----------------------------------------------------------

    # Problem 3: Write __eq__() and __str__().
   
    def __eq__(self, other):
        """An equality function. If the name, color, and number of contents are the same, then the two objects are equal."""
        return(self.name == other.name and self.color == other.color and len(self.contents) == len(other.contents))

    def __str__(self):
        """A function to print all of the attributes of a backpack."""
        return('Owner:\t\t'+self.name+'\nColor:\t\t'+ self.color+ '\nSize:\t\t' +                 str(len(self.contents))+ '\nMax Size:\t'+ str(self.max_size)+ '\nContents:\t'+ str(self.contents))


    def __add__(self, other):
        """Add the number of contents of each Backpack."""
        return len(self.contents) + len(other.contents)

    def __lt__(self, other):
        """Compare two backpacks. If 'self' has fewer contents
        than 'other', return True. Otherwise, return False.
        """
        return len(self.contents) < len(other.contents)


# An example of inheritance. You are not required to modify this class.
class Knapsack(Backpack):
    """A Knapsack object class. Inherits from the Backpack class.
    A knapsack is smaller than a backpack and can be tied closed.

    Attributes:
        name (str): the name of the knapsack's owner.
        color (str): the color of the knapsack.
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the knapsack is tied shut.
    """
    def __init__(self, name, color):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A knapsack only holds 3 item by default.

        Parameters:
            name (str): the name of the knapsack's owner.
            color (str): the color of the knapsack.
            max_size (int): the maximum number of items that can fit inside.
        """
        Backpack.__init__(self, name, color, max_size=3)
        self.closed = True

    def put(self, item):
        """If the knapsack is untied, use the Backpack.put() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.put(self, item)

    def take(self, item):
        """If the knapsack is untied, use the Backpack.take() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.take(self, item)

    def weight(self):
        """Calculate the weight of the knapsack by counting the length of the
        string representations of each item in the contents list.
        """
        return sum(len(str(item)) for item in self.contents)


# Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.
class Jetpack(Backpack):
    """A Jetpack object class. Has a name, color, max size, amount of fuel, and a list of contents.
    Attributes:
        name (str): the name of the backpack's owner.
        color (str): the color of the backpack
        max_size (int): the max size of the amount of items in the backpack 
        fuel (int): amount of fuel in the jetpack 
        contents (list): the contents of the backpack.
    """
    def __init__(self, name, color, max_size=2, fuel=10): 
        """Set the name, color, size, amount of fuel and initialize an empty list of contents.

        Parameters:
            name (str): the name of the backpack's owner.
            color (str): the color of the backpack
            max_size (int): the max amount of items that can fit in the backpack
            fuel (int): the amount of fuel in the jetpack
            contents (list): the contents of the backpack
        """    
        self.name = name  
        self.color = color
        self.max_size = max_size
        self.fuel = fuel              
        self.contents = []

    def fly(self, burn):
        """Takes an integer for an amount to burn in order to fly. If it exceeds the current fuel don't let the user blow the fuel
        """
        if burn > self.fuel:
            print('Not enough fuel!')
        else: 
            self.fuel -= burn

    def dump(self):
        """dumps the items in the backpack and the fuel"""
        Backpack.dump(self)
        self.fuel = 0


# Problem 4: Write a 'ComplexNumber' class.
class ComplexNumber():
    """A complex number class. Has a real and imaginary value with accompanying functions.
    Attributes:
    real (int): real number in the complex tuple
    imag (int): imaginary number in the complex tuple"""
    
    def __init__(self, real, imag):
        """Set the real and the imaginary number
        real (int): real number in the complex tuple
        imag (int): imaginary number in the complex tuple"""
        self.real = real
        self.imag = imag

    def conjugate(self):
        """Function to get the conjugate of the complex number"""
        con_imag = -self.imag
        return(ComplexNumber(self.real, con_imag))

    def __str__(self):
        """Overridden string function to print out a complex number in the form (a+bj)."""
        if self.imag >= 0:
            return('('+str(self.real)+'+'+str(self.imag)+'j)')
        else:
            return('('+str(self.real)+str(self.imag)+'j)')
    
    def __abs__(self):
        """Overridden absolute value function to get the magnitude of the complex number"""
        return(math.sqrt(self.real**2+self.imag**2))

    def __eq__(self, other):
        """Overridden equality function to see if two complex numbers are equal."""
        return(self.real == other.real and self.imag == other.imag)

    def __add__(self, other):
        """Overridden addition function to add two complex numbers"""
        return(ComplexNumber(self.real+other.real, self.imag+other.imag))

    def __sub__(self, other):
        """Overridden subtraction function to subtract two complex numbers"""
        return(ComplexNumber(self.real-other.real, self.imag-other.imag))

    def __mul__(self, other):
        """Overridden multiplication function to multiply two complex numbers"""
        real_number = self.real*other.real - self.imag*other.imag
        comp_number = self.real*other.imag + self.imag*other.real
        return(ComplexNumber(real_number,comp_number))

    def __truediv__(self, other):
        """Overridden true divide function to divide two complex numbers."""
        top = self * other.conjugate()
        bot = other * other.conjugate()
        real_number = top.real /bot.real
        imag_number = top.imag / bot.real
        return(ComplexNumber(real_number, imag_number))
