# solutions.py
"""Volume 2: One-Dimensional Optimization.
<Mark Rose>
<Section 2>
<1/30/19>
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linesearch
from autograd import numpy as anp
from autograd import grad

# Problem 1
def golden_section(f = lambda x: np.exp(x) - 4*x, a=0, b=3, tol=1e-5, maxiter=15):
    """Use the golden section search to minimize the unimodal function f.

    Parameters:
        f (function): A unimodal, scalar-valued function on [a,b].
        a (float): Left bound of the domain.
        b (float): Right bound of the domain.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    my_it = maxiter                                                 #Define the number of iterations variable and convergence boolean
    conv = False
    x0 = (a+b)/2                                                    #Set the initial minimizer approximation as the interval midpoint
    phi= (1+5**(1/2))/2
    for i in range(1, maxiter+1):                                   #Iterate only maxiter times at most
        c = (b-a)/phi
        a_t = b-c
        b_t = a+c
        if f(a_t) <= f(b_t):                                        #Get new boundaries for the search interval
            b = b_t
        else:
            a = a_t
        x1 = (a+b)/2                                                #Set the minimizer approximation as the interval midpoint
        if np.abs(x0-x1) < tol:
            conv = True
            my_it = i
            break                                                  #Stop iterating if the approximation stops changing enough                            
        x0 = x1
    return(x1, conv, my_it)
    raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def newton1d(df, d2f, x0, tol=1e-5, maxiter=15):
    """Use Newton's method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        d2f (function): The second derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    my_iter = maxiter                                                       #Initialize the iterations and convergence
    conv = False
    xold = x0
    for i in range(maxiter):
        xnew = xold - df(xold)/d2f(xold)                                      #Calculate the next value
        if (np.abs(xnew-xold) < tol):                                       #Check to see if it converged
            conv = True
            my_iter = i+1
            break
        xold = xnew
    return(xnew, conv, my_iter)                                             #Return the last calculated value, whether or not it converged, and the number of iterations that passed
    raise NotImplementedError("Problem 1 Incomplete")


# Problem 3
def secant1d(df, x0, x1, tol=1e-5, maxiter=15):
    """Use the secant method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        x1 (float): Another guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    my_iter = maxiter                                                       #Initialize the iterations and convergence
    conv = False
    for i in range(maxiter):
        first = df(x1)
        second = df(x0)
        xnew = (x0*first-x1*second)/(first-second)                          #Calculate the next value           
        if (np.abs(xnew-x1) < tol):                                         #Check to see if it converged
            conv = True
            my_iter = i+1
            break
        x0 = x1
        x1 = xnew
    return(xnew, conv, my_iter)                                             #Return the last caluculated value, convergence, and number of iterations
    raise NotImplementedError("Problem 1 Incomplete")


# Problem 4
def backtracking(f, Df, x, p, alpha=1, rho=.9, c=1e-4):
    """Implement the backtracking line search to find a step size that
    satisfies the Armijo condition.

    Parameters:
        f (function): A function f:R^n->R.
        Df (function): The first derivative (gradient) of f.
        x (float): The current approximation to the minimizer.
        p (float): The current search direction.
        alpha (float): A large initial step length.
        rho (float): Parameter in (0, 1).
        c (float): Parameter in (0, 1).

    Returns:
        alpha (float): Optimal step size.
    """
    if np.isscalar(x) == False:
        dfp = Df(x).T.dot(p)                                                          #Calculate these values only once
    else:
        dfp = Df(x)*p
    fx = f(x)
    while(f(x+alpha*p) > fx + c*alpha*dfp):
        alpha = rho*alpha                                                  #Scale down the alpha value by a constant rho
    return(alpha)
    raise NotImplementedError("Problem 1 Incomplete")
