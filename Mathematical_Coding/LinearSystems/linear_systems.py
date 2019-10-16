# linear_systems.py
"""Volume 1: Linear Systems.
<Mark Rose>
<Math 344 Section 2>
<10/6/18>
"""

import numpy as np
import scipy.linalg as la
import time
import matplotlib.pyplot as plt
from scipy import stats
from scipy import sparse
from scipy.sparse import linalg as spla


# Problem 1
def ref(A):
    """Reduce the square matrix A to REF. You may assume that A is invertible
    and that a 0 will never appear on the main diagonal. Avoid operating on
    entries that you know will be 0 before and after a row operation.

    Parameters:
        A ((n,n) ndarray): The square invertible matrix to be reduced.

    Returns:
        ((n,n) ndarray): The REF of A.
    """
    for i in range(len(A)):
        for j in range(i+1,len(A)):
            A[j,i:] -= (A[j,i]/A[i,i])*A[i,i:]                          #Subtract each row by a certain multiple of teh ones above it to make all 0's below the diagonal.
    return(A)
    raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def lu(A):
    """Compute the LU decomposition of the square matrix A. You may
    assume that the decomposition exists and requires no row swaps.

    Parameters:
        A ((n,n) ndarray): The matrix to decompose.

    Returns:
        L ((n,n) ndarray): The lower-triangular part of the decomposition.
        U ((n,n) ndarray): The upper-triangular part of the decomposition.
    """
    m,n = np.shape(A)
    U = A.copy()
    L = np.eye(m)
    for j in range(n):
        for i in range(j+1,m):                                          #Following the book algorithm to get the LU decomposition.
            L[i,j] = U[i,j]/U[j,j]                                      #Assume there are no row swaps
            U[i,j:] = U[i,j:] - L[i,j]*U[j,j:]
    return(L,U)
    
    raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def solve(A, b):
    """Use the LU decomposition and back substitution to solve the linear
    system Ax = b. You may again assume that no row swaps are required.

    Parameters:
        A ((n,n) ndarray)
        b ((n,) ndarray)

    Returns:
        x ((m,) ndarray): The solution to the linear system.
    """
      ## Find the LU decomposition from problem 2, create arrays for x,y values.
    L,U = lu(A)
    y = np.zeros(len(A))
    x = np.zeros(len(A))
    for j in range(len(A)):
        if j==0:
            y[j] = b[j]
        else:
            y[j] = b[j] - np.sum(np.dot(L[j,:j],y[:j].T))                           #Use the the algorithms to first solve for Ly=Pb 
    for i in range(len(A)-1,-1,-1):
        if i==len(A)-1:
            x[i] = 1/U[i,i]*y[i]                                                    #Then use the y found in the previous result to solve Ux=y
        else:
            x[i] = 1/U[i,i]*(y[i] - np.sum(np.dot(U[i,i+1:],x[i+1:])))
    return x                                                                        #return the resulting x vector
    raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def prob4():
    """Time different scipy.linalg functions for solving square linear systems.

    For various values of n, generate a random nxn matrix A and a random
    n-vector b using np.random.random(). Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Invert A with la.inv() and left-multiply the inverse to b.
        2. Use la.solve().
        3. Use la.lu_factor() and la.lu_solve() to solve the system with the
            LU decomposition.
        4. Use la.lu_factor() and la.lu_solve(), but only time la.lu_solve()
            (not the time it takes to do the factorization).

    Plot the system size n versus the execution times. Use log scales if
    needed.
    """
    n_list = [2,4,8,10,25,50,100,125,175,250]
    inv_times = []
    sol_times = []
    lu_times = []
    lusol_times = []
    for i in n_list:
        A = np.random.random(size = (i,i))
        b = np.random.random(size = i)
        
        time_inv = time.time()
        la.inv(A).dot(b)                                                                #Time every operation and save them in lists.
        time_inv = time.time()-time_inv
        inv_times.append(time_inv)
        
        time_solve = time.time()
        la.solve(A,b)
        time_solve = time.time()-time_solve
        sol_times.append(time_solve)
        
        time_lufac = time.time()
        L, P = la.lu_factor(A)
        x = la.lu_solve((L,P),b)
        time_lufac = time.time()-time_lufac
        lu_times.append(time_lufac)
        
        time_lusolve = time.time()
        x = la.lu_solve((L,P),b)
        time_lusolve = time.time()-time_lusolve
        lusol_times.append(time_lusolve)
    
    fig, axes= plt.subplots(1,2)
    axes[0].plot(n_list, inv_times,'b-o', label = 'Inverse Times')
    axes[0].plot(n_list, sol_times,'g-o', label = 'Solve Times')                    #Graph all of the times on one axes
    axes[0].plot(n_list, lu_times,'r-o', label = 'Solve with LU Decomposition Times')
    axes[0].plot(n_list, lusol_times,'c-o', label = 'Solve without LU Decomposition Times')
    axes[0].legend()
    axes[0].set_title('Scipy Execution Times')
    axes[0].set_xlabel('n')
    axes[0].set_ylabel('Seconds')
    axes[1].loglog(n_list, inv_times,'b-o', label = 'Inverse Times')
    axes[1].loglog(n_list, sol_times,'g-o', label = 'Solve Times')                   #Graph all of the log times on another axes
    axes[1].loglog(n_list, lu_times,'r-o', label = 'Solve with LU Decomposition Times')
    axes[1].loglog(n_list, lusol_times,'c-o', label = 'Solve without LU Decomposition Times')
    axes[1].legend()
    axes[1].set_title('Log Scipy Execution Time')
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('Seconds')
        
    plt.show()
    return
        
        
        
            
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def prob5(n):
    """Let I be the n × n identity matrix, and define
                    [B I        ]        [-4  1            ]
                    [I B I      ]        [ 1 -4  1         ]
                A = [  I . .    ]    B = [    1  .  .      ],
                    [      . . I]        [          .  .  1]
                    [        I B]        [             1 -4]
    where A is (n**2,n**2) and each block B is (n,n).
    Construct and returns A as a sparse matrix.

    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """
    B = sparse.diags([1,-4,1],[-1,0,1],shape=(n,n))
    D = sparse.block_diag([B]*n)                                                #Using the built in sparse matrix functions to build a block diagonal of the B matrix         
    D.setdiag(1,n)
    D.setdiag(1,-n)
    return D
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def prob6():
    """Time regular and sparse linear system solvers.

    For various values of n, generate the (n**2,n**2) matrix A described of
    prob5() and vector b of length n**2. Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Convert A to CSR format and use scipy.sparse.linalg.spsolve()
        2. Convert A to a NumPy array and use scipy.linalg.solve().

    In each experiment, only time how long it takes to solve the system (not
    how long it takes to convert A to the appropriate format). Plot the system
    size n**2 versus the execution times. As always, use log scales where
    appropriate and use a legend to label each line.
    """
    n = np.arange(2,10)**2
    csr_times = []
    np_times = []
    for j in n:
        A = prob5(j)
        b = np.random.random(j**2)
        Acsr = A.tocsr()
        start = time.time()                                                 #We time the different functions
        x = spla.spsolve(Acsr,b)
        csr_times.append(time.time() - start)
        Anp = np.asarray(A.toarray())
        start = time.time()
        x = la.solve(Anp,b)
        np_times.append(time.time() - start)

    plt.loglog(n**2,csr_times,'g.-',label="CSR Solve")
    plt.loglog(n**2,np_times,'r.-',label="Numpy Solve")                     #Use a log plot to plot each graph for numpy solve and CSR solve
    plt.title("CSR versus Numpy Solve Timing")
    plt.legend(loc="upper left")
    plt.show()
    raise NotImplementedError("Problem 6 Incomplete")
