# solutions.py
# iterative_solvers.py
"""Volume 1: Iterative Solvers.
<Mark Rose>
<Section 2>
<4/8/19>
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy import sparse


# Helper function
def diag_dom(n, num_entries=None):
    """Generate a strictly diagonally dominant (n, n) matrix.

    Parameters:
        n (int): The dimension of the system.
        num_entries (int): The number of nonzero values.
            Defaults to n^(3/2)-n.

    Returns:
        A ((n,n) ndarray): A (n, n) strictly diagonally dominant matrix.
    """
    if num_entries is None:
        num_entries = int(n**1.5) - n
    A = np.zeros((n,n))
    rows = np.random.choice(np.arange(0,n), size=num_entries)
    cols = np.random.choice(np.arange(0,n), size=num_entries)
    data = np.random.randint(-4, 4, size=num_entries)
    for i in range(num_entries):
        A[rows[i], cols[i]] = data[i]
    for i in range(n):
        A[i,i] = np.sum(np.abs(A[i])) + 1
    return A

# Problems 1 and 2
def jacobi(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Jacobi Method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        b ((n ,) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
    """
    #Initialize x and D
    abs_err = []
    x = np.zeros(np.shape(b))
    D = np.diag(A)
    #Get the new value using the equations
    for i in range(maxiter):
        d_inv = np.diag(1/D)
        x_new = x + d_inv @ (b-(A @ x))
        #Append the error 
        abs_err.append(la.norm(A @ x_new - b, ord = np.inf))
        if la.norm(x_new-x) < tol: 
            break
        x = x_new.copy()
    #If the plot=True plot the graph
    if plot == True:
        plt.semilogy(abs_err)
        plt.title('Convergence of Jacobi Method')
        plt.xlabel('Iteration')
        plt.ylabel('Absolute Error of Approximation')
        plt.show()
    return(x_new)
    raise NotImplementedError("Problem 1 Incomplete")


# Problem 3
def gauss_seidel(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Gauss-Seidel Method.

    Parameters:
        A ((n, n) ndarray): A square matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.
        plot (bool): If true, plot the convergence rate of the algorithm.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    #Initialize the vectors
    abs_err = []
    x = np.zeros(np.shape(b))
    x_new = x.copy()
    D = np.diag(A)
    D = 1/D
    #Update each of the values
    for j in range(maxiter):
        for i in range(len(b)):
            x_new[i] = x_new[i] + D[i] * (b[i] - (A[i] @ x_new))
        abs_err.append(la.norm(A @ x_new - b, ord = np.inf))
        if la.norm(x_new-x) < tol: 
            break
        x = x_new.copy()
    
    #Plot if it is true in the parameters
    if plot == True:
        plt.semilogy(abs_err)
        plt.title('Convergence of Gauss Seidel Method')
        plt.xlabel('Iteration')
        plt.ylabel('Absolute Error of Approximation')
        plt.show()
    return(x_new)
    raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def gauss_seidel_sparse(A, b, tol=1e-8, maxiter=100):
    """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
    Method.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse CSR matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    #Initialize the vecctors
    x = np.zeros(np.shape(b))
    x_new = x.copy()
    D = 1/ A.diagonal()
    for j in range(maxiter):
        for i in range(len(b)):
            #Use the special method to calculate Ax
            rowstart = A.indptr[i]
            rowend = A.indptr[i+1]
            x_new[i] = x_new[i] + D[i] * (b[i] - (A.data[rowstart:rowend] @ x_new[A.indices[rowstart:rowend]]))
        #Check to see if it converged
        if la.norm((x_new-x), ord = np.inf) < tol: 
            break
        x = x_new.copy()
    return(x_new)
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def sor(A, b, omega, tol=1e-8, maxiter=100):
    """Calculate the solution to the system Ax = b via Successive Over-
    Relaxation.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse matrix.
        b ((n, ) Numpy Array): A vector of length n.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    #Initialize the vectors
    converged=False
    x = np.zeros(np.shape(b))
    x_new = x.copy()
    #Include omega
    D = omega/ A.diagonal()
    for j in range(maxiter):
        for i in range(len(b)):
            #Use the special method to get Ax
            rowstart = A.indptr[i]
            rowend = A.indptr[i+1]
            x_new[i] = x_new[i] + (D[i] * (b[i] - A.data[rowstart:rowend] @ x_new[A.indices[rowstart:rowend]]))
        #Check the norm
        if la.norm(x_new-x) < tol: 
            converged=True
            break
        x = x_new.copy()
    return(x_new,converged, j+1)
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def hot_plate(n, omega, tol=1e-8, maxiter=100, plot=False):
    """Generate the system Au = b and then solve it using sor().
    If show is True, visualize the solution with a heatmap.

    Parameters:
        n (int): Determines the size of A and b.
            A is (n^2, n^2) and b is one-dimensional with n^2 entries.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The iteration tolerance.
        maxiter (int): The maximum number of iterations.
        plot (bool): Whether or not to visualize the solution.

    Returns:
        ((n^2,) ndarray): The 1-D solution vector u of the system Au = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of computed iterations in SOR.
    """
    #Create hte matrix A
    I = sparse.diags([1,1],[-n,n],shape=(n**2,n**2))
    B = sparse.diags([1,-4,1],[-1,0,1],shape=(n,n))
    A=sparse.block_diag([B]*n)+I
    #Create the matrix B
    b= np.zeros(n**2)
    b[::n],b[n-1::n] = -100,-100
    u, conv, num_it = sor(A,b,omega,tol,maxiter)
    #Plot if it is True
    if plot == True:
        plt.pcolormesh(np.reshape(u,(n,n)), cmap='coolwarm')
        plt.show()
    return(u, conv, num_it)
    
    
    raise NotImplementedError("Problem 6 Incomplete")


# Problem 7
def prob7():
    """Run hot_plate() with omega = 1, 1.05, 1.1, ..., 1.9, 1.95, tol=1e-2,
    and maxiter = 1000 with A and b generated with n=20. Plot the iterations
    computed as a function of omega.
    """
    #Initiazlize the omegas
    my_iter = []
    omega = np.linspace(1,2,21)[:-1]
    for i in omega:
        #Get the iterations for each omega
        u, conv, num_it = hot_plate(20,i,tol=1e-2,maxiter=1000)
        my_iter.append(num_it)
    #Plot the graph and label it
    plt.plot(omega, my_iter)
    plt.title("Relaxation Factor's Effect on Convergence")
    plt.xlabel("Relaxation Factor")
    plt.ylabel("Number of Iterations")
    plt.show()
    #Return the best omega value
    return(omega[np.argmin(my_iter)])
    raise NotImplementedError("Problem 7 Incomplete")



