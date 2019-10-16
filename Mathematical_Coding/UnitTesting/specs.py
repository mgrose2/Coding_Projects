# specs.py
"""Python Essentials: Unit Testing.
<Mark Rose>
<Section 2>
<10/22/18>
"""

def add(a, b):
    """Add two numbers."""
    return a + b

def divide(a, b):
    """Divide two numbers, raising an error if the second number is zero."""
    if b == 0:
        raise ZeroDivisionError("second input cannot be zero")
    return a / b


# Problem 1
def smallest_factor(n):
    """Return the smallest prime factor of the positive integer n."""
    if type(n) != int or n <= 0:
        return None
    if n == 1: return 1
    elif int(n**.5) == 2:
        if n % 2 == 0:
            return 2
    for i in range(2, int(n**.5)):
        if n % i == 0: return i
    return n


# Problem 2
def month_length(month, leap_year=False):
    """Return the number of days in the given month."""
    if month in {"September", "April", "June", "November"}:
        return 30
    elif month in {"January", "March", "May", "July",
                        "August", "October", "December"}:
        return 31
    if month == "February":
        if not leap_year:
            return 28
        else:
            return 29
    else:
        return None


# Problem 3
def operate(a, b, oper):
    """Apply an arithmetic operation to a and b."""
    if type(oper) is not str:
        raise TypeError("oper must be a string")
    elif oper == '+':
        return a + b
    elif oper == '-':
        return a - b
    elif oper == '*':
        return a * b
    elif oper == '/':
        if b == 0:
            raise ZeroDivisionError("division by zero is undefined")
        return a / b
    raise ValueError("oper must be one of '+', '/', '-', or '*'")


# Problem 4
class Fraction(object):
    """Reduced fraction class with integer numerator and denominator."""
    def __init__(self, numerator, denominator):
        if denominator == 0:
            raise ZeroDivisionError("denominator cannot be zero")
        elif type(numerator) is not int or type(denominator) is not int:
            raise TypeError("numerator and denominator must be integers")

        def gcd(a,b):
            while b != 0:
                a, b = b, a % b
            return a
        common_factor = gcd(numerator, denominator)
        self.numer = numerator // common_factor
        self.denom = denominator // common_factor

    def __str__(self):
        if self.denom != 1:
            return "{}/{}".format(self.numer, self.denom)
        else:
            return str(self.numer)

    def __float__(self):
        return self.numer / self.denom

    def __eq__(self, other):
        if type(other) is Fraction:
            return self.numer==other.numer and self.denom==other.denom
        else:
            return float(self) == other

    def __add__(self, other):
        return Fraction(self.numer*other.denom + self.denom*other.numer,
                                                        self.denom*other.denom)
    def __sub__(self, other):
        return Fraction(self.numer*other.denom - self.denom*other.numer,
                                                        self.denom*other.denom)
    def __mul__(self, other):
        return Fraction(self.numer*other.numer, self.denom*other.denom)

    def __truediv__(self, other):
        if self.denom*other.numer == 0:
            raise ZeroDivisionError("cannot divide by zero")
        return Fraction(self.numer*other.denom, self.denom*other.numer)


# Problem 6
def count_sets(cards):
    """Return the number of sets in the provided Set hand.

    Parameters:
        cards (list(str)) a list of twelve cards as 4-bit integers in
        base 3 as strings, such as ["1022", "1122", ..., "1020"].
    Returns:
        (int) The number of sets in the hand.
    Raises:
        ValueError: if the list does not contain a valid Set hand, meaning
            - there are not exactly 12 cards,
            - the cards are not all unique,
            - one or more cards does not have exactly 4 digits, or
            - one or more cards has a character other than 0, 1, or 2.
    """
    
    if len(cards) != 12:						
        raise ValueError("list must have exactly 12 cards")                                 #check the number of cards
    
    for i in range(12):							                                            #check that all cards are unique
        for j in range(i+1, 12):
            if cards[i] == cards[j]:
                raise ValueError("the cards must all be unique")
    for i in range(12):							                                            #check that all have 4 digits
        if len(cards[i]) != 4:
            raise ValueError("all cards must have exactly 4 digits")
    for i in range(12):							                                            #check that all have values 0,1,2
        for j in range(4):
            if cards[i][j] != '0' and cards[i][j] != '1' and cards[i][j] != '2':
                raise ValueError("all characters must be 0, 1, or 2")
                
    total_sets = 0								                                            #count the number of sets
    for a in range(12):
        for b in range(a+1, 12):
            for c in range(b+1, 12):
                if is_set(cards[a],cards[b],cards[c]):
                    total_sets += 1
    return total_sets
                    

def is_set(a, b, c):
    """Determine if the cards a, b, and c constitute a set.

    Parameters:
        a, b, c (str): string representations of 4-bit integers in base 3.
            For example, "1022", "1122", and "1020" (which is not a set).
    Returns:
        True if a, b, and c form a set, meaning the ith digit of a, b,
            and c are either the same or all different for i=1,2,3,4.
        False if a, b, and c do not form a set.
    """
    if a == b or b == c or a == c:						                                    #verify all cards are unique
        raise ValueError("the cards must all be unique")
    if len(a) != 4 or len(b) != 4 or len(c) != 4:		                                     #verify all have 4 digits
        raise ValueError("all cards must have exactly 4 digits")
    check = [a,b,c]
    for j in range(3):									                                     #verify all have values 0,1,2
        for i in range(4):
            if (check[j][i] != '0' and check[j][i] != '1' and check[j][i] != '2'):
                raise ValueError("all characters must be 0, 1, or 2")
    
    for i in range(4):									                                     #verify that all equal or all are different
        all_equal = (a[i] == b[i] == c[i])
        all_diff = (a[i] != b[i] and a[i] != c[i] and b[i] != c[i])
        
        if all_equal is not True and all_diff is not True:
            return False
    return True
