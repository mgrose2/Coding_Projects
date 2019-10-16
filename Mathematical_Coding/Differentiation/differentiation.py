# solutions.py
"""Volume 1: Differentiation.
<Mark Rose>
<Section 2>
<1/21/19>
"""
import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
import math
from autograd import numpy as anp       # Use autograd's version of NumPy.
from autograd import grad
from autograd import elementwise_grad
import time

# Problem 1
def prob1():
    """Return the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy."""
    x = sy.symbols('x')                                                     #Define the variables
    f = (sy.sin(x)+1)**(sy.sin(sy.cos(x)))                                  #Define f
    deriv = sy.diff(f, x)                                                   #Calculate the function
    f = sy.lambdify(x, f, 'numpy')
    f_der = sy.lambdify(x, deriv,'numpy')                                   #Create two callable functions
    """   
    dom = np.linspace(-np.pi, np.pi,100)
    plt.plot(dom, f(dom), label='Original Function', color='r')             #Plot the original function and the derivative
    plt.plot(dom, f_der(dom), label='Function Derivative', color='b')
    plt.legend()
    ax = plt.gca()
    ax.spines["bottom"].set_position('zero')                                #Set the plots close to the x-axis
    plt.show()
    """
    return(f_der)
    raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def fdq1(f, x, h=1e-5):
    """Calculate the first order forward difference quotient of f at x."""
    return((f(x+h)-f(x))/h)                                                #Calculate the first order forward differencce quotient of f
    raise NotImplementedError("Problem 2 Incomplete")

def fdq2(f, x, h=1e-5):
    """Calculate the second order forward difference quotient of f at x."""
    return((-3*f(x)+4*f(x+h)-f(x+2*h))/(2*h))                              #Calculate the second order forward differencce quotient of f
    raise NotImplementedError("Problem 2 Incomplete")

def bdq1(f, x, h=1e-5):
    """Calculate the first order backward difference quotient of f at x."""
    return((f(x)-f(x-h))/h)                                                #Calculate the first order backward differencce quotient of f
    raise NotImplementedError("Problem 2 Incomplete")

def bdq2(f, x, h=1e-5):
    """Calculate the second order backward difference quotient of f at x."""
    return((3*f(x)-4*f(x-h)+f(x-2*h))/(2*h))                               #Calculate the second order backward differencce quotient of f
    raise NotImplementedError("Problem 2 Incomplete")

def cdq2(f, x, h=1e-5):
    """Calculate the second order centered difference quotient of f at x."""
    return((f(x+h)-f(x-h))/(2*h))                                         #Calculate the second order centered differencce quotient of f
    raise NotImplementedError("Problem 2 Incomplete")

def cdq4(f, x, h=1e-5):
    """Calculate the fourth order centered difference quotient of f at x."""
    return((f(x-2*h)-8*f(x-h)+8*f(x+h)-f(x+2*h))/(12*h))                  #Calculate the fourth order centered differencce quotient of f
    raise NotImplementedError("Problem 2 Incomplete")

    

# Problem 3
def prob3(x0):
    """Let f(x) = (sin(x) + 1)^(sin(cos(x))). Use prob1() to calculate the
    exact value of f'(x0). Then use fdq1(), fdq2(), bdq1(), bdq2(), cdq1(),
    and cdq2() to approximate f'(x0) for h=10^-8, 10^-7, ..., 10^-1, 1.
    Track the absolute error for each trial, then plot the absolute error
    against h on a log-log scale.

    Parameters:
        x0 (float): The point where the derivative is being approximated.
    """
    x = sy.symbols('x')                                                   #Define the symbols
    f = (sy.sin(x)+1)**(sy.sin(sy.cos(x)))
    f = sy.lambdify(x,f,'numpy')                                          #create f
    f_derv = prob1()
    val = f_derv(x0)
    hs = np.logspace(-8,0,9)                                              #Create the hs
    order_one_for = []                                                    #Create a bunch of error lists
    order_two_for = []
    order_one_back = []
    order_two_back = []
    order_two_cent = []
    order_four_cent = []
    for h in hs:
        order_one_for.append(np.abs(val-fdq1(f,x0,h)))                   #Calculate each error
        order_two_for.append(np.abs(val-fdq2(f,x0,h)))
        order_one_back.append(np.abs(val-bdq1(f,x0,h)))
        order_two_back.append(np.abs(val-bdq2(f,x0,h)))
        order_two_cent.append(np.abs(val-cdq2(f,x0,h)))
        order_four_cent.append(np.abs(val-cdq4(f,x0,h)))
    plt.loglog(hs, order_one_for, label = 'Order 1 Forward' )            #plot each error on a loglog graph
    plt.loglog(hs, order_two_for, label = 'Order 2 Forward' )
    plt.loglog(hs, order_one_back, label = 'Order 1 Backward' )
    plt.loglog(hs, order_two_back, label = 'Order 2 Backward' )
    plt.loglog(hs, order_two_cent, label = 'Order 2 Centered' )
    plt.loglog(hs, order_four_cent, label = 'Order 4 Centered' )
    plt.xlabel('h')
    plt.ylabel('Absolute Error')                                        #set the labels
    plt.title('Derivative Error')
    plt.legend()
    plt.show()
    return

    raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def prob4():
    """The radar stations A and B, separated by the distance 500m, track a
    plane C by recording the angles alpha and beta at one-second intervals.
    Your goal, back at air traffic control, is to determine the speed of the
    plane.

    Successive readings for alpha and beta at integer times t=7,8,...,14
    are stored in the file plane.npy. Each row in the array represents a
    different reading; the columns are the observation time t, the angle
    alpha (in degrees), and the angle beta (also in degrees), in that order.
    The Cartesian coordinates of the plane can be calculated from the angles
    alpha and beta as follows.

    x(alpha, beta) = a tan(beta) / (tan(beta) - tan(alpha))
    y(alpha, beta) = (a tan(beta) tan(alpha)) / (tan(beta) - tan(alpha))

    Load the data, convert alpha and beta to radians, then compute the
    coordinates x(t) and y(t) at each given t. Approximate x'(t) and y'(t)
    using a forward difference quotient for t=7, a backward difference
    quotient for t=14, and a centered difference quotient for t=8,9,...,13.
    Return the values of the speed at each t.
    """ 
    data = np.load('plane.npy')                                 #load the data
    data[:,1:]= np.deg2rad(data[:,1:])                          #turn the degrees to radians
    al = data[:,1]
    bet = data[:,2]
    x = 500*np.tan(bet)/(np.tan(bet)-np.tan(al))                #calculate the x and y
    y = 500*np.tan(bet)*np.tan(al)/(np.tan(bet)-np.tan(al))
    x_prime = []
    y_prime = []
    x_prime.append(x[1]-x[0])                                   #use forward order 1 at the beginning
    y_prime.append(y[1]-y[0])
    for i in range(1,7):
        x_prime.append((x[i+1]-x[i-1])/2)                       #Then Use the central order 2
        y_prime.append((y[i+1]-y[i-1])/2)
    x_prime.append(x[7]-x[6])                                   #Lastly use the backward order 1
    y_prime.append(y[7]-y[6])
    x_prime = np.array(x_prime)
    y_prime = np.array(y_prime)
    return((x_prime**2+y_prime**2)**(1/2))                      #Return the speed
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def jacobian_cdq2(f, x, h=1e-5):
    """Approximate the Jacobian matrix of f:R^n->R^m at x using the second
    order centered difference quotient.

    Parameters:
        f (function): the multidimensional function to differentiate.
            Accepts a NumPy (n,) ndarray and returns an (m,) ndarray.
            For example, f(x,y) = [x+y, xy**2] could be implemented as follows.
            >>> f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2])
        x ((n,) ndarray): the point in R^n at which to compute the Jacobian.
        h (float): the step size in the finite difference quotient.

    Returns:
        ((m,n) ndarray) the Jacobian matrix of f at x.
    """

    jacobian = np.ndarray((len(f(x)), len(x)))		#initialize jacobian
    I = np.identity(len(x))							#initialize identity matrix
    
    for i in range(len(x)):							#calculate jacobian using cdq2
        jacobian[:,i] = (f(x+h*I[:,i]) - f(x-h*I[:,i])) / (2*h)
    
    return jacobian 
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def cheb_poly(x, n):
    """Compute the nth Chebyshev polynomial at x.

    Parameters:
        x (autograd.ndarray): the points to evaluate T_n(x) at.
        n (int): The degree of the polynomial.
    """
    if n == 0:
    	return anp.ones_like(x)                        #Return an array of ones
    if n == 1:
        return x
    else:
        return 2*x*cheb_poly(x, n-1) - cheb_poly(x, n-2)
    raise NotImplementedError("Problem 6 Incomplete")

def prob6():
    """Use Autograd and cheb_poly() to create a function for the derivative
    of the Chebyshev polynomials, and use that function to plot the derivatives
    over the domain [-1,1] for n=0,1,2,3,4.
    """
    dT = elementwise_grad(cheb_poly)		#Calculate T'n(x)
    
    x = np.linspace(-1, 1, 1000)
    for n in range(5):						#graph the derivative for n=0,1,2,3,4
        print(n)
        plt.plot(x, dT(x, n), color="blue")
        plt.title("Derivative of Chebyshev for n=" + str(n))
        plt.show()
        
    return
    raise NotImplementedError("Problem 6 Incomplete")


# Problem 7
def prob7(N=200):
    """Let f(x) = (sin(x) + 1)^sin(cos(x)). Perform the following experiment N
    times:

        1. Choose a random value x0.
        2. Use prob1() to calculate the “exact” value of f′(x0). Time how long
            the entire process takes, including calling prob1() (each
            iteration).
        3. Time how long it takes to get an approximation of f'(x0) using
            cdq4(). Record the absolute error of the approximation.
        4. Time how long it takes to get an approximation of f'(x0) using
            Autograd (calling grad() every time). Record the absolute error of
            the approximation.

    Plot the computation times versus the absolute errors on a log-log plot
    with different colors for SymPy, the difference quotient, and Autograd.
    For SymPy, assume an absolute error of 1e-18.
    """
    x = sy.symbols('x')							#define the symbols and function
    fun = lambda x: (anp.sin(x) + 1)**anp.sin(anp.cos(x))
    
    domain = np.random.randn(N)					#choose N random values
    time_prob1 = []								#initialize arrays for time and error
    time_cdq4 = []
    time_autograd = []
    error1 = [1*10**-18 for i in range(N)]
    error2 = []
    error3 = []
    
    for x in domain:							#iterate through N random values
        start = time.time()						#time prob1()
        exact = prob1()(x)
        end = time.time()
        time_prob1.append(end-start)

        start = time.time()						#time cdq4()
        y=cdq4(fun, x)
        end = time.time()
        time_cdq4.append(end-start)
        error2.append(np.abs(y-exact))
        
        start = time.time()						#time Autograd
        df = grad(fun)(x)
        end = time.time()
        time_autograd.append(end-start)
        error3.append(np.abs(df-exact))
    											#graph    
    plt.loglog(time_prob1, error1, "o", color="orange", label="SymPy")
    plt.loglog(time_cdq4, error2, "o", color="green", label="Difference Quotients")
    plt.loglog(time_autograd, error3, "o", color="blue", label="Autograd")
    plt.xlabel("Computation Time (seconds)")
    plt.ylabel("Absolute Error")
    plt.title("Times and Errors of Derivatives")
    plt.legend()
    plt.show()
    return
    raise NotImplementedError("Problem 7 Incomplete")
