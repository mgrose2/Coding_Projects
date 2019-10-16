# condition_stability.py
"""Volume 1: Conditioning and Stability.
<Mark Rose>
<Section 2>
<2/4/19>
"""

import numpy as np
import sympy as sy
import scipy.linalg as la
import matplotlib.pyplot as plt

# Problem 1
def matrix_cond(A):
    """Calculate the condition number of A with respect to the 2-norm."""
    U, s, V = la.svd(A)                                                     #Get the singular values by calculating the svd
    sin_max = np.max(s)                                                     #Get the max and min singular values
    sin_min = np.min(s)
    if (sin_min == 0):                                                      #Return infinity if the minimum is 0
        return(np.inf)
    else:
        return(sin_max/sin_min)
    raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def prob2():
    """Randomly perturb the coefficients of the Wilkinson polynomial by
    replacing each coefficient c_i with c_i*r_i, where r_i is drawn from a
    normal distribution centered at 1 with standard deviation 1e-10.
    Plot the roots of 100 such experiments in a single figure, along with the
    roots of the unperturbed polynomial w(x).

    Returns:
        (float) The average absolute condition number.
        (float) The average relative condition number.
    """
    w_roots = np.arange(1, 21)

    # Get the exact Wilkinson polynomial coefficients using SymPy.
    x, i = sy.symbols('x i')                    
    w = sy.poly_from_expr(sy.product(x-i, (i, 1, 20)))[0]                                      
    w_coeffs = np.array(w.all_coeffs())
    #Create my coefficient list
    abs_k = []
    rel_k = []
    for i in range(100):
        h = np.random.normal(loc = 1, scale = 1e-10, size = (21))
        #Perturb the coefficients very slightly
        new_coeffs = h*w_coeffs
        #Use NumPy to compute the roots of the perturbed polynomial
        new_roots = np.roots(np.poly1d(new_coeffs))
        w_roots = np.sort(w_roots)
        new_roots = np.sort(new_roots)
        #Extimate the condition numbers in the infinity norm
        abs_k.append(la.norm(new_roots-w_roots, np.inf)/la.norm(h, np.inf))
        rel_k = (abs_k[i]*la.norm(w_coeffs, np.inf)/la.norm(w_roots, np.inf))
        if i == 99:
            plt.scatter(np.real(new_roots), np.imag(new_roots), s=.1,color='black',label='Perturbed')
        else:
            plt.scatter(np.real(new_roots), np.imag(new_roots), s=.1,color='black')
    #Plot the original roots
    plt.scatter(np.real(w_roots), np.imag(w_roots), s=10,color='b', label = 'Original')
    plt.xlabel('Real Axis')
    plt.ylabel('Imaginary Axis')
    plt.legend()
    plt.show()
    return(np.mean(abs_k), np.mean(rel_k))
    
    raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def eig_cond(A):
    """Approximate the condition numbers of the eigenvalue problem at A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) The absolute condition number of the eigenvalue problem at A.
        (float) The relative condition number of the eigenvalue problem at A.
    """
    reals = np.random.normal(0,1e-10,A.shape)
    imags = np.random.normal(0,1e-10,A.shape)
    H = reals +1j*imags
    #Create an A hat with the a pertubation H
    A_hat = A+H
    lamb = la.eigvals(A)
    lamb_hat = la.eigvals(A_hat)
    #Using the eigenvalues get the condition numbers
    abs_cond = la.norm(lamb-lamb_hat, ord=2)/la.norm(H, ord=2)
    rel_cond = la.norm(A,ord=2)/la.norm(lamb,ord=2)*abs_cond
    return(abs_cond, rel_cond)
    raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def prob4(domain=[-100, 100, -100, 100], res=50):
    """Create a grid [x_min, x_max] x [y_min, y_max] with the given resolution. For each
    entry (x,y) in the grid, find the relative condition number of the
    eigenvalue problem, using the matrix   [[1, x], [y, 1]]  as the input.
    Use plt.pcolormesh() to plot the condition number over the entire grid.

    Parameters:
        domain ([x_min, x_max, y_min, y_max]):
        res (int): number of points along each edge of the grid.
    """
    #Create a resolution domain
    x_res = np.linspace(domain[0],domain[1], res)
    y_res = np.linspace(domain[2],domain[3], res)
    my_map = np.zeros((res,res))
    for i, x in enumerate(x_res):
        for j, y in enumerate(y_res):
            #Use the previous function to compute the condition numbers
            my_map[i,j] = eig_cond(np.array([[1,x],[y,1]]))[1]
    #Plot on a gray colormesh
    plt.pcolormesh(x_res, y_res, my_map, cmap='gray_r')
    plt.colorbar()
    plt.show()
    return
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def prob5(n):
    """Approximate the data from "stability_data.npy" on the interval [0,1]
    with a least squares polynomial of degree n. Solve the least squares
    problem using the normal equation and the QR decomposition, then compare
    the two solutions by plotting them together with the data. Return
    the mean squared error of both solutions, ||Ax-b||_2.

    Parameters:
        n (int): The degree of the polynomial to be used in the approximation.

    Returns:
        (float): The forward error using the normal equations.
        (float): The forward error using the QR decomposition.
    """
    #Get the data and create a vander matrix
    xk,yk = np.load('stability_data.npy').T
    A = np.vander(xk, n+1)
    #Solve using the inverse method and the QR triangluar method
    c1 = la.inv(A.T.dot(A)).dot(A.T).dot(yk)
    q, r = la.qr(A, mode='economic')
    c2 = la.solve_triangular(r, q.T.dot(yk))
    plt.scatter(xk, yk, s = 1, color='black')
    #Plot both equations by their polynomials
    plt.plot(xk, np.polyval(c1, xk), color = 'blue', label = 'Normal Equations')
    plt.plot(xk, np.polyval(c2, xk), color = 'orange', label = 'QR Solver')
    plt.ylim(0,25)
    plt.legend()
    plt.show()
    #Return the forward error of both approximations
    return(la.norm(A.dot(c1)-yk, ord=2), la.norm(A.dot(c2)-yk, ord=2))
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def prob6():
    """For n = 5, 10, ..., 50, compute the integral I(n) using SymPy (the
    true values) and the subfactorial formula (may or may not be correct).
    Plot the relative forward error of the subfactorial formula for each
    value of n. Use a log scale for the y-axis.
    """
    #Create a range of ns
    ns = np.arange(5,51,5)
    x = sy.symbols('x')
    int_meth = []
    subfac_meth = []
    for n in ns:
        #Calculate the integral using sympy and the subfactorial formula
        int_meth.append(float(sy.integrate(x**n*sy.exp(x-1), (x,0,1))))
        subfac_meth.append((-1)**n*(sy.subfactorial(n)-sy.factorial(n)/np.exp(1)))
    my_val = np.array(int_meth)-np.array(subfac_meth)
    #Calculate the Error
    print(int_meth)
    print(subfac_meth)
    my_err = np.abs(my_val)/np.abs(np.array(int_meth))
    #Plot the error by the value of n with a y log-scale
    plt.plot(ns, my_err)
    plt.xlabel('n')
    plt.ylabel('Error')
    plt.title('Error by Value of N')
    plt.yscale('log')
    plt.show()
    return
    raise NotImplementedError("Problem 6 Incomplete")

#Prob6() Answer: The subfactorial formula is not a stable method for computing the integral with large values of n as the forward error exceeds 100 very quickly.
