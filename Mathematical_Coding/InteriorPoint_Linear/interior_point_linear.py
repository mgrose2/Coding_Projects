# interior_point_linear.py
"""Volume 2: Interior Point for Linear Programs.
<Mark Rose>
<Section 2>
<3/21/19>
"""

import numpy as np
from scipy import linalg as la
from scipy.stats import linregress
from matplotlib import pyplot as plt


# Auxiliary Functions ---------------------------------------------------------
def startingPoint(A, b, c):
    """Calculate an initial guess to the solution of the linear program
    min c^T x, Ax = b, x>=0.
    Reference: Nocedal and Wright, p. 410.
    """
    # Calculate x, lam, mu of minimal norm satisfying both
    # the primal and dual constraints.
    B = la.inv(A @ A.T)
    x = A.T @ B @ b
    lam = B @ A @ c
    mu = c - (A.T @ lam)

    # Perturb x and s so they are nonnegative.
    dx = max((-3./2)*x.min(), 0)
    dmu = max((-3./2)*mu.min(), 0)
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    # Perturb x and mu so they are not too small and not too dissimilar.
    dx = .5*(x*mu).sum()/mu.sum()
    dmu = .5*(x*mu).sum()/x.sum()
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    return x, lam, mu

# Use this linear program generator to test your interior point method.
def randomLP(m):
    """Generate a 'square' linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add slack variables.
    Parameters:
        m -- positive integer: the number of desired constraints
             and the dimension of space in which to optimize.
    Returns:
        A -- array of shape (m,n).
        b -- array of shape (m,).
        c -- array of shape (n,).
        x -- the solution to the LP.
    """
    n = m
    A = np.random.random((m,n))*20 - 10
    A[A[:,-1]<0] *= -1
    x = np.random.random(n)*10
    b = A.dot(x)
    c = A.sum(axis=0)/float(n)
    return A, b, -c, x

# This random linear program generator is more general than the first.
def randomLP2(m,n):
    """Generate a linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add
    slack variables to convert it into the above form.
    Parameters:
        m -- positive integer >= n, number of desired constraints
        n -- dimension of space in which to optimize
    Returns:
        A -- array of shape (m,n+m)
        b -- array of shape (m,)
        c -- array of shape (n+m,), with m trailing 0s
        v -- the solution to the LP
    """
    A = np.random.random((m,n))*20 - 10
    A[A[:,-1]<0] *= -1
    v = np.random.random(n)*10
    k = n
    b = np.zeros(m)
    b[:k] = A[:k,:].dot(v)
    b[k:] = A[k:,:].dot(v) + np.random.random(m-k)*10
    c = np.zeros(n+m)
    c[:n] = A[:k,:].sum(axis=0)/k
    A = np.hstack((A, np.eye(m)))
    return A, b, -c, v


# Problems --------------------------------------------------------------------
def interiorPoint(A, b, c, niter=20, tol=1e-16, verbose=False):
    """Solve the linear program min c^T x, Ax = b, x>=0
    using an Interior Point method.

    Parameters:
        A ((m,n) ndarray): Equality constraint matrix with full row rank.
        b ((m, ) ndarray): Equality constraint vector.
        c ((n, ) ndarray): Linear objective function coefficients.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    #Create a function to return the almost-linear function F
    m,n = A.shape
    
    F = lambda x, l, mu: np.hstack((A.T@l + mu - c, A@x - b, np.diag(mu)@x))
    
    #calculate the Df matrix
    def DF(X,lmbda,µ):
        m,n = A.shape
        M = np.diag(µ)
        
        #Stack all the rows
        row1 = np.hstack([np.zeros((n,n)),A.T,np.eye(n)])
        row2 = np.hstack([A,np.zeros((m,m)),np.zeros((m,n))])
        row3 = np.hstack([M,np.zeros((n,m)),np.diag(X)])
        return np.vstack([row1,row2,row3])
    
    #Create a function for the search direction
    def solve(X,lmbda,µ,sigma = 0.1):
        m,n = A.shape
        lu, piv = la.lu_factor(DF(X,lmbda,µ))
        v = (1/n)*X@µ

        #figure out b
        b = -F(X,lmbda,µ) + np.hstack([np.zeros(n+m),sigma*v*np.ones(n)])
        return la.lu_solve((lu,piv),b)
        
    def update(X,lmbda,µ,sigma = 0.1):

        #Assign all of the delta values
        m,n = A.shape
        delta = solve(X,lmbda,µ)
        deltaU = delta[n+m:]
        
        #Use various masksk
        deltaU1 = deltaU[deltaU < 0]
        deltaL = delta[n:n+m]
        deltaL1 = deltaL[deltaL < 0]
        deltaX = delta[:n]
        deltaX1 = deltaX[deltaX < 0]
        
        #Get the correct values
        minU = np.min(-µ[deltaU < 0]/deltaU1)
        amax = np.min([1.,minU])
        minX = np.min(-X[deltaX < 0]/deltaX1)
        dmax = np.min([1.,minX])

        #final a and d values we want
        a = np.min([1.,0.95*amax])
        d = np.min([1.,0.95*dmax])
        X = X + d*deltaX
        lmbda += a*deltaL
        µ += a*deltaU

        #return new values
        return X,lmbda,µ

    X,lmbda,µ = startingPoint(A,b,c)

    #iterate
    for i in range(niter):
        v = (1/n)*X@µ
        if v < tol:
            break
        X,lmbda,µ = update(X,lmbda,µ)

    return X, c@X


def leastAbsoluteDeviations(filename='simdata.txt'):
    """Generate and show the plot requested in the lab."""
    #get data
    data = np.loadtxt(filename)

    #change the data 
    m = data.shape[0]
    n = data.shape[1] - 1
    c = np.zeros(3*m + 2*(n + 1))
    c[:m] = 1
    y = np.empty(2*m)
    y[::2] = -data[:, 0]
    y[1::2] = data[:, 0]
    x = data[:, 1:]

    #follow example as to how to set up A
    A = np.ones((2*m, 3*m + 2*(n + 1)))
    A[::2, :m] = np.eye(m)
    A[1::2, :m] = np.eye(m)
    A[::2, m:m+n] = -x
    A[1::2, m:m+n] = x
    A[::2, m+n:m+2*n] = x
    A[1::2, m+n:m+2*n] = -x
    A[::2, m+2*n] = -1
    A[1::2, m+2*n+1] = -1
    A[:, m+2*n+2:] = -np.eye(2*m, 2*m)

    #We will now use the interior points
    sol = interiorPoint(A, y, c, niter=10)[0]
    beta = sol[m:m+n] - sol[m+n:m+2*n]
    b = sol[m+2*n] - sol[m+2*n+1]

    #Graph the linear regression line
    slope, intercept = linregress(data[:,1], data[:,0])[:2]
    domain = np.linspace(0,10,200)
    plt.plot(domain, domain*slope + intercept, label = "Least Squares Approximation")
    plt.scatter(data[:,1],data[:,0],s = 4,c = "black")
    plt.plot(domain,beta*domain+b, label = "LAD approximation")
    
    #Label and title everything
    plt.title("SimData Line Fit Regressions")
    plt.xlabel('Explanatory Variables')
    plt.ylabel('Response Variables')
    plt.legend()
    plt.show()
    return
