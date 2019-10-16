# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
<Mark Rose>
<Section 2>
<11/1/8>
"""

# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la
import cmath

# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    Q,R = la.qr(A, mode='economic')                         #we use economic for the reduced qr decomp
    return(la.solve_triangular(R, Q.T.dot(b)))              #returns x_hat, the solution to Rx = Q.T.dot(b)
    raise NotImplementedError("Problem 1 Incomplete")

# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    my_array = np.load('housing.npy')                          #open the file and save the contents
    A = np.ones(np.shape(my_array))
    A[:,0] = my_array[:,0]
    b = my_array[:,1]
    my_points = least_squares(A,b)                              #Plot the original points
    plt.scatter(A[:,0], b, color='r')
    plt.plot(A[:,0],A.dot(my_points), color='b')                #Plot x values and y values according to the equation y=ax+b where a and b are my_points
    plt.xlabel('Year (Starting at Year 2000)')
    plt.ylabel('Price Index ($)')
    plt.title('House Price Index')
    plt.show()
    return
    raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    my_array = np.load('housing.npy')
    my_sol = []
    my_x = np.linspace(0,16, num = 100)                          #Getting an array of values that will make the polynomials smooth
    fig, axes= plt.subplots(2,2)
    for i in range(4,14,3):
        A= np.vander(my_array[:,0], i)                           #Make the necessary A and b arrays for the solution
        b = my_array[:,1]
        x = np.poly1d(la.lstsq(A,b)[0])                          #Append a callable polynomial
        my_sol.append(x)
    axes[0,0].scatter(my_array[:,0],b,color='r')                 #Scatter all of the original points
    axes[0,0].plot(my_x, my_sol[0](my_x), color='b')             #Call the polynomial on the range of x values and plot it
    axes[0,0].set_xlabel('Year (Starting at 2000)')              #Set all labels and titles for each axes
    axes[0,0].set_ylabel('Price Index ($)')
    axes[0,0].set_title('Housing Price Index 3rd Degree Polynomial Fit')
    axes[0,1].scatter(my_array[:,0],b,color='r')
    axes[0,1].plot(my_x, my_sol[1](my_x), color='b')
    axes[0,1].set_xlabel('Year (Starting at 2000)')
    axes[0,1].set_ylabel('Price Index ($)')
    axes[0,1].set_title('Housing Price Index 6th Degree Polynomial Fit')
    axes[1,0].scatter(my_array[:,0],b,color='r')
    axes[1,0].plot(my_x, my_sol[2](my_x), color='b')
    axes[1,0].set_xlabel('Year (Starting at 2000)')
    axes[1,0].set_ylabel('Price Index ($)')
    axes[1,0].set_title('Housing Price Index 9th Degree Polynomial Fit')
    axes[1,1].scatter(my_array[:,0],b,color='r')
    axes[1,1].plot(my_x, my_sol[3](my_x), color='b')
    axes[1,1].set_xlabel('Year (Starting at 2000)')
    axes[1,1].set_ylabel('Price Index ($)')
    axes[1,1].set_title('Housing Price Index 12th Degree Polynomial Fit')
    plt.show()
    return
    raise NotImplementedError("Problem 3 Incomplete")


def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    my_array = np.load('ellipse.npy')                                     #Load the data into an array
    x = my_array[:,0]
    y = my_array[:,1]
    A = np.column_stack((x**2, x, x*y, y, y**2))                          #Make one array for all the parameters
    b = np.ones(len(x))
    a,b,c,d,e = la.lstsq(A,b)[0]                                          #Solve the equation and get a,b,c,d,e
    plt.scatter(x,y, color='r')
    plot_ellipse(a,b,c,d,e)                                               #Scatter original points and use premade plot_ellipse function to plot the ellipse with the parameters
    plt.show()
    return
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    m, n = np.shape(A)
    x_n = np.random.random(n)
    x_n = x_n/la.norm(x_n)                                                #innitialize a random x_n vector
    for k in range(1,N):
        x_k = A.dot(x_n)/la.norm(A.dot(x_n))                              #get the next vector and normalize it
        if (la.norm(x_k-x_n) < tol):                                      #If ||xk-xn|| < tolerance, then return the eigenvalue and its corresponding eigenvector
            return(x_k.T.dot(A.dot(x_k)), x_k)
        x_n = x_k                           
    return(x_n.T.dot(A.dot(x_n)),x_n)                                     #Return the eigenvalue by computing x_nTAx_n and return the corresponding eigenvector x_n
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    m, n = np.shape(A)
    S = la.hessenberg(A)                                                   #Put A in upper Hessenberg form
    for k in range(N):
        Q,R = la.qr(S)                                                     #Get the QR decomposition of Ak
        S = R.dot(Q)                                                       #Recombine Rk and Qk into Ak+1
    eigs = []                                                              #Initialize an empty list of eigenvalues
    i = 0
    while i < n:
        if i == n-1 or abs(S[i+1][i] < tol):
            eigs.append(S[i][i])
        else:
            b = -1*(S[i,i] + S[i+1,i+1])
            c = (S[i,i]*S[i+1,i+1]-S[i,i+1]*S[i+1,i])
            eigs.append((-1*b + cmath.sqrt(b**2-4*c)) / 2)                  #If 2X2 get the two eigenvalues by using the quadratic formula
            eigs.append((-1*b - cmath.sqrt(b**2-4*c)) / 2)
            i+=1
        i+=1
    return(np.array(eigs))                                                  #Returns an array of the list of eigenvalues.
            
    raise NotImplementedError("Problem 6 Incomplete")
