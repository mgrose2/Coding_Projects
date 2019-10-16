# solutions.py
"""Volume 2: Gaussian Quadrature.
<Mark Rose>
<Section 2>
<1/23/19>
"""
from numpy.linalg import eig as eig
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.pyplot as plt

class GaussianQuadrature:
    """Class for integrating functions on arbitrary intervals using Gaussian
    quadrature with the Legendre polynomials or the Chebyshev polynomials.
    """
    # Problems 1 and 3
    def __init__(self, n, polytype="legendre"):
        """Calculate and store the n points and weights corresponding to the
        specified class of orthogonal polynomial (Problem 3). Also store the
        inverse weight function w(x)^{-1} = 1 / w(x).

        Parameters:
            n (int): Number of points and weights to use in the quadrature.
            polytype (string): The class of orthogonal polynomials to use in
                the quadrature. Must be either 'legendre' or 'chebyshev'.

        Raises:
            ValueError: if polytype is not 'legendre' or 'chebyshev'.
        """
        if polytype != "legendre" and polytype != 'chebyshev':
            raise ValueError('The polyomial type must be legendre or chebyshev')          #Raise a value error if the wrong type is inputted
        self.polytype = polytype
        self.n = n
        if polytype == 'legendre':
            self.recip = lambda x: 1                                                      #Set the reciprocating function
        else:
            self.recip = lambda x: (1-x**2)**(1/2)
        self.points, self.weights = self.points_weights(n)                               #Set the ponts ad weights from the points_weights function
        return
        raise NotImplementedError("Problem 1 Incomplete")

    # Problem 2
    def points_weights(self, n):
        """Calculate the n points and weights for Gaussian quadrature.

        Parameters:
            n (int): The number of desired points and weights.

        Returns:
            points ((n,) ndarray): The sampling points for the quadrature.
            weights ((n,) ndarray): The weights corresponding to the points.
        """
        J = np.zeros((n,n))                                                       #Initialize an nxn matrix
        if self.polytype == 'legendre':                                           
            for i in range(n-1):
                J[i, i+1] = ((i+1)**2/(4*(i+1)**2-1))**(1/2)                      #Use the equation to set each beta value in the Matrix
                J[i+1,i] = ((i+1)**2/(4*(i+1)**2-1))**(1/2)
        else:
            for i in range(n-1):
                if i == 0:
                    J[i, i+1] = (1/2)**(1/2)                                      #Use the chebyshev equation to set the beta values
                    J[i+1,i] = (1/2)**(1/2)
                else:
                    J[i, i+1]= 1/2
                    J[i+1,i] = 1/2
        points, eig_vecs = eig(J)                                                #Get the eigenvalues and eigenvectors
        weights = np.zeros_like(points)#Calculate the legrende and chebyshev approximations for various values of n
        for i in range(n):
            if self.polytype == 'legendre':
                weights[i] = 2*eig_vecs[0,i]**2                                 #Get the weights depending on the polynomial type
            else:
                weights[i] = np.pi*eig_vecs[0,i]**2
        return(points, weights)                                                 #Return the points and weights
        raise NotImplementedError("Problem 2 Incomplete")

    # Problem 3
    def basic(self, f):
        """Approximate the integral of a f on the interval [-1,1]."""
        return(np.sum(f(self.points)*self.recip(self.points)*self.weights))      #Return the sum of using the sample points and weights found previous part
        raise NotImplementedError("Problem 3 Incomplete")

    # Problem 4
    def integrate(self, f, a, b):#Calculate the legrende and chebyshev approximations for various values of n
        """Approximate the integral of a function on the interval [a,b].

        Parameters:
            f (function): Callable function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.

        Returns:
            (float): Approximate value of the integral.
        """
        h = f((b-a)/2*self.points + (a+b)/2)                                    #Calculate h at the points
        bott = self.recip(self.points)                                          #Calculate w inverse at the points
        return((b-a)/2*np.sum(h*bott*self.weights))                             #Return the integral by the formula given
        raise NotImplementedError("Problem 4 Incomplete")

    # Problem 6.
    def integrate2d(self, f, a1, b1, a2, b2):
        """Approximate the integral of the two-dimensional function f on
        the interval [a1,b1]x[a2,b2].

        Parameters:
            f (function): A function to integrate that takes two parameters.
            a1 (float): Lower bound of integration in the x-dimension.
            b1 (float): Upper bound of integration in the x-dimension.
            a2 (float): Lower bound of integration in the y-dimension.
            b2 (float): Upper bound of integration in the y-dimension.

        Returns:
            (float): Approximate value of the integral.
        """
        ab_cons1 = (b1-a1)/2                                                       #Define each of the ab constants beforehand
        ab_cons2 = (a1+b1)/2
        ab_cons3 = (b2-a2)/2
        ab_cons4 = (a2+b2)/2
        ab_cons5 = (b1-a1)*(b2-a2)/4
        total = 0
        for i in range(self.n):                                                    #Define two for loops for the summation
            for j in range(self.n):
                total+=self.weights[i]*self.weights[j]*f(ab_cons1*self.points[i]+ab_cons2,  ab_cons3*self.points[j]+ab_cons4)*self.recip(i)*self.recip(j)     #Use the given equation to calculate the double sum
        return(ab_cons5*total)                                                    #return the summation times the last constant
        raise NotImplementedError("Problem 6 Incomplete")


# Problem 5
def prob5():
    """Use scipy.stats to calculate the "exact" value F of the integral of
    f(x) = (1/sqrt(2 pi))e^((-x^2)/2) from -3 to 2. Then repeat the following
    experiment for n = 5, 10, 15, ..., 50.
        1. Use the GaussianQuadrature class with the Legendre polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
        2. Use the GaussianQuadrature class with the Chebyshev polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
    Plot the errors against the number of points and weights n, using a log
    scale for the y-axis. Finally, plot a horizontal line showing the error of
    scipy.integrate.quad() (which doesn’t depend on n).
    """
    f = lambda x: 1/(2*np.pi)**(1/2)*np.exp(-x**2/2)                               #Define a function f
    val = norm.cdf(2) - norm.cdf(-3)                                               #Get the 'exact' answer   
    my_l_est = []
    my_c_est = []
    for n in np.arange(5,51,5):
        quad_l_func = GaussianQuadrature(n)                                        #Create two objects, one for chebyshev and one for legendre
        quad_c_func = GaussianQuadrature(n, 'chebyshev')
        my_l_est.append(quad_l_func.integrate(f,-3,2))                             #Calculate the legendre and chebyshev approximations for various values of n
        my_c_est.append(quad_c_func.integrate(f,-3,2))
    my_l_err = (np.abs(np.array(my_l_est)-val))
    my_c_err = (np.abs(np.array(my_c_est)-val))
    quad_err = np.abs(quad(f,-3,2)[0]-val)*np.ones_like(my_c_err)              #Calculate my errors and the error from the scipy quad function
    plt.plot(np.arange(5,51,5),my_l_err, label='Legendre Error')                   #Graph all of the errors
    plt.plot(np.arange(5,51,5), my_c_err, label = 'Chebyshev Error')
    plt.plot(np.arange(5,51,5), quad_err, label = 'Scipy Function Error')
    plt.xlabel('N')
    plt.ylabel('Error')                                                            #Label, title, and change the scale of the graph
    plt.title("Error of Integration by N")
    plt.yscale('log')
    plt.legend()
    plt.show()
    return
    raise NotImplementedError("Problem 5 Incomplete")
