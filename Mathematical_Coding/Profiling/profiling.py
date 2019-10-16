# profiling.py
"""Python Essentials: Profiling.
<Mark Rose>
<Volume 1>
<1/8/18>
"""

# Note: for problems 1-4, you need only implement the second function listed.
# For example, you need to write max_path_fast(), but keep max_path() unchanged
# so you can do a before-and-after comparison.

import numpy as np
import math
from numba import jit
import matplotlib.pyplot as plt
import time


# Problem 1
def max_path(filename="triangle.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    def path_sum(r, c, total):
        """Recursively compute the max sum of the path starting in row r
        and column c, given the current total.
        """
        total += data[r][c]
        if r == len(data) - 1:          # Base case.
            return total
        else:                           # Recursive case.
            return max(path_sum(r+1, c,   total),   # Next row, same column
                       path_sum(r+1, c+1, total))   # Next row, next column

    return path_sum(0, 0, 0)            # Start the recursion from the top.

def max_path_fast(filename="triangle_large.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    data.reverse()
    for counter in range(len(data[1:])):
        row_up = np.array(data[counter])
        row_ev = row_up[:-1]
        row_odd = row_up[1:]
        row_one = np.array(data[counter+1])+row_ev              #Add the numbers to the top left
        row_two = np.array(data[counter+1])+row_odd             #Add the numbers to the top right
        mask = (row_one-row_two)>=0                             #create some masks to use the correct values
        unmask = (row_one-row_two) < 0
        new_row = row_one*mask
        right_part = unmask*row_two
        new_row = new_row+right_part                           #Make the new row
        data[counter+1] = new_row
    return(data[-1][0])                                       #return the largest sum



# Problem 2
def primes(N):
    """Compute the first N primes."""
    primes_list = []
    current = 2
    while len(primes_list) < N:
        isprime = True
        for i in range(2, current):     # Check for nontrivial divisors.
            if current % i == 0:
                isprime = False
        if isprime:
            primes_list.append(current)
        current += 1
    return primes_list

def primes_fast(N):
    """Compute the first N primes."""
    primes_list = [2]                          #Initialize the list
    current = 3
    while len(primes_list) < N:
        isprime = True
        root =current**(1/2)                #Use the fact that the primes need to be less than half the square root of n
        for i in primes_list:     # Check for nontrivial divisors.
            if current % i == 0:
                isprime = False
                break                         #Break if the number isn't prime
            elif i > root:
                break
        if isprime:
            primes_list.append(current)
        current += 2                         #Go up by twos to take away evens
    return primes_list



# Problem 3
def nearest_column(A, x):
    """Find the index of the column of A that is closest to x.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    distances = []
    for j in range(A.shape[1]):
        distances.append(np.linalg.norm(A[:,j] - x))
    return np.argmin(distances)

def nearest_column_fast(A, x):
    """Find the index of the column of A that is closest in norm to x.
    Refrain from using any loops or list comprehensions.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    return(np.argmin(np.linalg.norm((A.T-x).T, axis=0)))                      #Use the norm for the colums and take the minimum value



# Problem 4
def name_scores(filename="names.txt"):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    total = 0
    for i in range(len(names)):
        name_value = 0
        for j in range(len(names[i])):
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for k in range(len(alphabet)):
                if names[i][j] == alphabet[k]:
                    letter_value = k + 1
            name_value += letter_value
        total += (names.index(names[i]) + 1) * name_value
    return total

def name_scores_fast(filename='names.txt'):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    total = 0                                                                                             #initialize the total to zero
    my_d = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,                                          #create a hash table
                'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17,
                'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25, 'Z':26}
    for counter, name in enumerate(names):
        for j in name:
            total+=(counter+1)*my_d[j]                                                                    #add each value times its position
    return(total)
        


# Problem 5
def fibonacci():
    """Yield the terms of the Fibonacci sequence with F_1 = F_2 = 1."""
    first = 1
    second = 1                            #Initialize the first and second values
    while True:
        new_val = first + second          #create the next fibonzcci number
        first = second
        second = new_val
        yield new_val                     #yield that fibonacci number
        
    

def fibonacci_digits(N=1000):
    """Return the index of the first term in the Fibonacci sequence with
    N digits.

    Returns:
        (int): The index.
    """
    if N ==1:
        return 0                              #base case if n = 1
    for i, x in enumerate(fibonacci()):       #Use the generator
        if len(str(x)) == N:
            return(i+2)                       #return the correct index


# Problem 6
def prime_sieve(N):
    """Yield all primes that are less than N."""
    numbers = list(range(2, N))                                   #Create list of integers
    yield 2
    while numbers:
        new_list = list(l for l in numbers if l % numbers[0])   #Remove all muptiples of first integer
        numbers = new_list					
        if len(numbers) > 0:				                    #Repeat until there are no integers left
            yield numbers[0]


# Problem 7
def matrix_power(A, n):
    """Compute A^n, the n-th power of the matrix A."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

def matrix_power_numba(A, n):
    """Compute A^n, the n-th power of the matrix A, with Numba optimization."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):							#Repeat for n powers
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]      #calculate total 
                temporary_array[j] = total
            product[i] = temporary_array	
    return product

def prob7(n=10):
    """Time matrix_power(), matrix_power_numba(), and np.linalg.matrix_power()
    on square matrices of increasing size. Plot the times versus the size.
    """
    m = np.array([2**2, 2**3, 2**4, 2**5, 2**6, 2**7])
    times1 = []                                  #time the matrix_power() function 
    for i in m:
        A = np.random.random((i,i))
        start = time.time()
        matrix_power(A,n)
        end = time.time()
        times1.append(end - start)
    
    #time the matrix_power_numba() function
    times2 = []
    for i in m:
        A = np.random.random((i,i))
        start = time.time()
        matrix_power_numba(A,n)
        end = time.time()
        times2.append(end-start)
        
    #time the numpy matrix power function
    times3 = []
    for i in m:
        A = np.random.random((i,i))
        start = time.time()
        np.linalg.matrix_power(A,n)
        end = time.time()
        times3.append(end-start)
    
    #graph the different function times
    plt.loglog(m, times1, 'b.-', linewidth=2, markersize=15, label="Matrix Power")
    plt.loglog(m, times2, '.-', color='orange', linewidth=2, markersize=15, label="Numba")
    plt.loglog(m, times3, 'm.-', linewidth=2, markersize=15, label="NumPy")
    plt.xlabel("m", fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    plt.legend(loc = "upper left")
    
    plt.show()
