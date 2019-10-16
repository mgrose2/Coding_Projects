# newtons_method.py
"""Volume 1: Newton's Method.
<Mark Rose>
<Section 2>
<1/28/18>
"""
import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
import scipy.linalg as la

# Problems 1, 3, and 5
def newton(f, x0, Df, tol=1e-5, maxiter=15, alpha=1.):
    """Use Newton's method to approximate a zero of the function f.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.
        alpha (float): Backtracking scalar (Problem 3).

    Returns:
        (float or ndarray): The approximation for a zero of f.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    my_iter = maxiter                                                       #Initialize the iterations and convergence
    conv = False
    xold = x0
    for i in range(maxiter):
        if np.isscalar(x0) == True:
            xnew = xold - alpha*f(xold)/Df(xold)                                #Calculate the next value with alpha discount factor
        else:
            xnew = xold - alpha*la.solve(Df(xold), f(xold))
        if (la.norm(xnew-xold) < tol):                                       #Check to see if it converged
            conv = True
            my_iter = i+1
            break
        xold = xnew
    return(xnew, conv, my_iter)                                             #Return the last calculated value, whether or not it converged, and the number of iterations that passed
    raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def prob2(N1, N2, P1, P2):
    """Use Newton's method to solve for the constant r that satisfies

                P1[(1+r)**N1 - 1] = P2[1 - (1+r)**(-N2)].

    Use r_0 = 0.1 for the initial guess.

    Parameters:
        P1 (float): Amount of money deposited into account at the beginning of
            years 1, 2, ..., N1.
        P2 (float): Amount of money withdrawn at the beginning of years N1+1,
            N1+2, ..., N1+N2.
        N1 (int): Number of years money is deposited.
        N2 (int): Number of years money is withdrawn.

    Returns:
        (float): the value of r that satisfies the equation.
    """
    x = sy.symbols('x')                                                      #Define the x symbol
    f = P1*((1+x)**N1-1) - P2*(1-(1+x)**(-N2))                               #Create the function
    deriv = sy.diff(f,x)                                                     #Get the derivative
    f = sy.lambdify(x,f)
    Df = sy.lambdify(x,deriv)
    r = .1                                                                   #Initialize a close r
    val, conv, it = newton(f, r, Df)
    return(val)                                                              #Return the value
    raise NotImplementedError("Problem 2 Incomplete")


# Problem 4
def optimal_alpha(f, x0, Df, tol=1e-5, maxiter=15):
    """Run Newton's method for various values of alpha in (0,1].
    Plot the alpha value against the number of iterations until convergence.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): a value for alpha that results in the lowest number of
            iterations.
    """
    my_iter = []
    my_dom = np.linspace(.001,1,100)                                         #Create an Alpha domain
    for i in my_dom:
        vals = newton(f,x0, Df, tol, maxiter, i)                             #Append the amount of iterations it took to converge
        my_iter.append(vals[2])
    plt.plot(my_dom, np.array(my_iter))                                      #Plot the data, title, x and y labels
    plt.title('Alpha Value Test')
    plt.xlabel('Alpha Values')
    plt.ylabel('Iterations Until Convergence')
    plt.show()
    return
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 6
def prob6():
    """Consider the following Bioremediation system.

                              5xy − x(1 + y) = 0
                        −xy + (1 − y)(1 + y) = 0

    Find an initial point such that Newton’s method converges to either
    (0,1) or (0,−1) with alpha = 1, and to (3.75, .25) with alpha = 0.55.
    Return the intial point as a 1-D NumPy array with 2 entries.
    """
    xdomain = np.linspace(-.25, 0, 100)				#initialize x and y domains
    ydomain = np.linspace(0, .25, 100)
    
    s1 = np.array([0,1])							#save solutions
    s2 = np.array([0,-1])
    s3 = np.array([3.75, .25])
    
    												#calculate f and df
    f = lambda x: np.array([5*x[0]*x[1] - x[0]*(1+x[1]), -x[0]*x[1] + (1-x[1])*(1+x[1])])
    df = lambda x: np.array([[5*x[1] - (1+x[1]), 5*x[0] - x[0]], [-x[1], -x[0] + -2*x[1]]])
    
    for i in range(100):
        for j in range(100):						#iterate and calculate x0
            x0 = newton(f, np.array([xdomain[i], ydomain[j]]), df)[0]
        
            if np.allclose(x0, s1) or np.allclose(x0, s2):		#if x0 is a solution, repeat for alpha=.55
                x00 = newton(f, np.array([xdomain[i], ydomain[j]]), df, tol=1e-5, maxiter=15, alpha=.55)[0]
            
                if np.allclose(x00, s3):			#check if x00 is the solution
                    return np.array([xdomain[i], ydomain[j]])
    return

    raise NotImplementedError("Problem 6 Incomplete")


# Problem 7
def plot_basins(f, Df, zeros, domain, res=1000, iters=15):
    """Plot the basins of attraction of f on the complex plane.

    Parameters:
        f (function): A function from C to C.
        Df (function): The derivative of f, a function from C to C.
        zeros (ndarray): A 1-D array of the zeros of f.
        domain ([r_min, r_max, i_min, i_max]): A list of scalars that define
            the window limits and grid domain for the plot.
        res (int): A scalar that determines the resolution of the plot.
            The visualized grid has shape (res, res).
        iters (int): The exact number of times to iterate Newton's method.
    """
    x_real = np.linspace(domain[0], domain[1], res)		#Real parts
    x_imag = np.linspace(domain[2], domain[3], res)		#Imaginary parts
    X_real, X_imag = np.meshgrid(x_real, x_imag)
    X_0 = X_real + 1j*X_imag 							#Combine real and imaginary parts
    
    for i in range(iters):								#Newton's method
        X_1 = X_0 - f(X_0)/Df(X_0)
        X_0 = X_1
        
    Y = np.copy(X_1)									#Compute Y
    for i in range(res):
        for j in range(res):
            Y[i,j] = np.argmin(np.abs(zeros - Y[i,j]))
    Y = Y.real
            
    plt.pcolormesh(X_real, X_imag, Y, cmap="brg")		#Plot
    plt.show()
    return
    raise NotImplementedError("Problem 7 Incomplete")
