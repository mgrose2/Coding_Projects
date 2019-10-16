# solutions.py
"""Volume 2: Polynomial Interpolation.
<Mark Rose>
<Section 2>
<1/17/19>
"""

import numpy as np
from scipy.interpolate import BarycentricInterpolator
import scipy.linalg as la
import matplotlib.pyplot as plt
from numpy.fft import fft

# Problem 1
def lagrange(xint, yint, points):
    """Find an interpolating polynomial of lowest degree through the points
    (xint, yint) using the Lagrange method and evaluate that polynomial at
    the specified points.

    Parameters:
        xint ((n,) ndarray): x values to be interpolated.
        yint ((n,) ndarray): y values to be interpolated.
        points((m,) ndarray): x values at which to evaluate the polynomial.

    Returns:
        ((m,) ndarray): The value of the polynomial at the specified points.
    """
    my_vals = np.zeros(len(points))                                            #initialize the end values, numerator and denoinator
    numer = np.zeros(len(xint))
    denom = np.zeros(len(xint))
    for i, x in enumerate(points):
        for count, val in enumerate(xint):
            prod = (val-xint)                                                  #calculate the product in the denominator
            denom[count] = np.prod(prod[prod!=0])
            top = (x-xint)
            numer[count] = np.prod(top[top!=top[count]])                      #Calculate the product in the numerator
        my_vals[i] = np.sum((numer/denom)*yint)                               #Calculate the new value
    return(my_vals)
    raise NotImplementedError("Problem 1 Incomplete")


# Problems 2 and 3
class Barycentric:
    """Class for performing Barycentric Lagrange interpolation.

    Attributes:
        w ((n,) ndarray): Array of Barycentric weights.
        n (int): Number of interpolation points.
        x ((n,) ndarray): x values of interpolating points.
        y ((n,) ndarray): y values of interpolating points.
    """

    def __init__(self, xint, yint):
        """Calculate the Barycentric weights using initial interpolating points.

        Parameters:
            xint ((n,) ndarray): x values of interpolating points.
            yint ((n,) ndarray): y values of interpolating points.
        """
        my_vec = np.zeros(len(xint))                                            #Initialize the weight vector
        for i in range(len(xint)):
            my_new = xint.copy()                                                #create a copy
            my_new= np.delete(my_new,i)                                         #delete the ith value
            my_vec[i] = (float(np.prod(xint[i]-my_new)))**(-1)                  #take the product
        self.w = my_vec                                                         #set the weights x's and y's as attributes
        self.x = xint
        self.y = yint

    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """
        my_y = np.zeros(len(points))                                           #initialize the y values 
        for j in range(len(points)):
            bot = points[j]-self.x                                             #Calculate the bottom of the equation 
            val = np.sum(self.w/bot*self.y)/(np.sum(self.w/bot))               #Calculate the whole value
            my_y[j] = val
        return(my_y)                                                           #Return the calculated values
        raise NotImplementedError("Problem 2 Incomplete")

    # Problem 3
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        try:                                                            #I use the try except method to help with the case when xint is just one value
            for count, val in enumerate(xint):                          #Enumerate through the xints
                self.w = self.w/(self.x-val)                            #update existing weights
                new_weight = 1/np.prod(val-self.x)                      #calculate the new weights
                self.w = np.hstack((self.w, new_weight))
                self.x = np.hstack((self.x, val))                       #update the rest of the attributes
                self.y = np.hstack((self.y, yint[count])) 
        except TypeError:
            val = xint                                                  #If x-int is just a one element array, do the same thing as above but without iterating through
            self.w = self.w/(self.x-val)
            new_weight = 1/np.prod(val-self.x)
            self.w = np.hstack((self.w, new_weight))
            self.x = np.hstack((self.x, val))
            self.y = np.hstack((self.y, yint)) 
        return
        raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def prob4():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """
    
    x = np.linspace(-1,1,400)                                               #Create the domain with 400 equally spaced values
    n = 2**(np.arange(2,9))
    f= lambda x:1/(1+25*x**2)                                               #Initialize Runge's function
    my_err = []
    my_cheb_err = []                                                        #Initialize the barycentric error list and the chebyshev error list
    for i in n:
        pts = np.linspace(-1,1,i)                                           #The number of values to interpolate
        y = f(x)
        poly = BarycentricInterpolator(pts, f(pts))
        err = la.norm(f(x)-poly(x), ord=np.inf)                             #Get the barycentric error
        my_err.append(err)
        cheb_pts = np.cos(np.arange(i+1)*np.pi/i)                           #Get the chebyshev points
        cheb_poly = BarycentricInterpolator(cheb_pts,(1/(1+25*cheb_pts**2)))
        cheb_err = la.norm(f(x)-cheb_poly(x), ord=np.inf)                   #Calculate the chebyshev polynomial and its error
        my_cheb_err.append(cheb_err)
    plt.title("Barycentric and  Chebyshev Error")
    plt.xlabel('N Interpolating Points')
    plt.loglog(n, my_err, label='Equally Spaced Points Error', basex =2)    #Graph both errors on a loglog graph
    plt.loglog(n, my_cheb_err, label='Chebyshev Extremal Points Error', basex=2)
    plt.legend()
    plt.show()
    return
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def chebyshev_coeffs(f, n):
    """Obtain the Chebyshev coefficients of a polynomial that interpolates
    the function f at n points.

    Parameters:
        f (function): Function to be interpolated.
        n (int): Number of points at which to interpolate.

    Returns:
        coeffs ((n+1,) ndarray): Chebyshev coefficients for the interpolating polynomial.
    """
    y= np.cos((np.pi*np.arange(2*n)) / n)                                
    samples = f(y)                                                       #Calculate the samples
    
    coeffs = np.real(fft(samples))[:n+1]/ n                              #Get the real portion of the discrete fourier transform of the samples
    coeffs[0] = coeffs[0]/2
    coeffs[n] = coeffs[n]/2                                              #Make sure to fix the gamma k constants
    
    return coeffs
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def prob6(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """
   
    data = np.load('airdata.npy')                                       #Load the data
    dom = np.linspace(0,366-1/24,len(data))                             #Create a common domain
    fx = lambda a,b,n: .5*(a+b + (b-a)*np.cos(np.arange(n+1)*np.pi/n))
    a,b = 0, 366-1/24                                                   #Give some bounds
    domain = np.linspace(0,b,8784)
    points = fx(a,b,n)
    temp = np.abs(points-domain.reshape(8784,1))
    temp2 = np.argmin(temp, axis=0)                                     #Find the closest points to the extremizers
    poly = Barycentric(domain[temp2], data[temp2])                      #Create a polynomial using the barycentric method
    fig, axes = plt.subplots(1,2)
    axes[0].plot(dom, data)                                             #Plot the domain with the data
    axes[0].set_ylim(-20,100)
    axes[1].plot(dom, poly(dom))                                        #Plot the domain with the approximation
    axes[1].set_ylim(-20,100)
    plt.show()
    return
    raise NotImplementedError("Problem 6 Incomplete")
