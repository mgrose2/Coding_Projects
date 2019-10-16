# interior_point_quadratic.py
"""Volume 2: Interior Point for Quadratic Programs.
<Mark Rose>
<Section 2>
<4/1/19>
"""
from cvxopt import matrix, solvers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
from scipy.sparse import spdiags


def startingPoint(G, c, A, b, guess):
    """
    Obtain an appropriate initial point for solving the QP
    .5 x^T Gx + x^T c s.t. Ax >= b.
    Parameters:
        G -- symmetric positive semidefinite matrix shape (n,n)
        c -- array of length n
        A -- constraint matrix shape (m,n)
        b -- array of length m
        guess -- a tuple of arrays (x, y, mu) of lengths n, m, and m, resp.
    Returns:
        a tuple of arrays (x0, y0, l0) of lengths n, m, and m, resp.
    """
    m,n = A.shape
    x0, y0, l0 = guess

    # Initialize linear system
    N = np.zeros((n+m+m, n+m+m))
    N[:n,:n] = G
    N[:n, n+m:] = -A.T
    N[n:n+m, :n] = A
    N[n:n+m, n:n+m] = -np.eye(m)
    N[n+m:, n:n+m] = np.diag(l0)
    N[n+m:, n+m:] = np.diag(y0)
    rhs = np.empty(n+m+m)
    rhs[:n] = -(G.dot(x0) - A.T.dot(l0)+c)
    rhs[n:n+m] = -(A.dot(x0) - y0 - b)
    rhs[n+m:] = -(y0*l0)

    sol = la.solve(N, rhs)
    dx = sol[:n]
    dy = sol[n:n+m]
    dl = sol[n+m:]

    y0 = np.maximum(1, np.abs(y0 + dy))
    l0 = np.maximum(1, np.abs(l0+dl))

    return x0, y0, l0


# Problems 1-2
def qInteriorPoint(Q, c, A, b, guess, niter=20, tol=1e-16, verbose=False):
    """Solve the Quadratic program min .5 x^T Q x +  c^T x, Ax >= b
    using an Interior Point method.

    Parameters:
        Q ((n,n) ndarray): Positive semidefinite objective matrix.
        c ((n, ) ndarray): linear objective vector.
        A ((m,n) ndarray): Inequality constraint matrix.
        b ((m, ) ndarray): Inequality constraint vector.
        guess (3-tuple of arrays of lengths n, m, and m): Initial guesses for
            the solution x and lagrange multipliers y and eta, respectively.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    #first set things as float, just in case
    m,n = A.shape
    Q = Q.astype(np.float64)
    A = A.astype(np.float64)
    b = b.astype(np.float64)
    c = c.astype(np.float64)
    x, y, u = startingPoint(Q,c,A,b,guess)

    #determine F function
    F = lambda x, y, µ: np.hstack([Q@x - A.T@µ + c, A@x - y - b, np.diag(y)@np.diag(µ)@np.ones(m)])

    #function to make DF
    def DF(x,y,µ):
        m,n = A.shape
        M = np.diag(µ)
        row1 = np.hstack([Q, np.zeros((n,m)), -1*A.T])
        row2 = np.hstack([A, -1*np.eye(m), np.zeros((m,m))])
        row3 = np.hstack([np.zeros((m,n)), M, np.diag(y)])
        return np.vstack([row1,row2,row3])

    #solver function
    def solve(x,y,µ,sigma = 0.1):
        m,n = A.shape
        lu, piv = la.lu_factor(DF(x,y,µ))
        v = (1/m)*y.T@µ

        #figure out b
        b = -F(x,y,µ) + np.hstack([np.zeros(n+m),sigma*v*np.ones(m)])
        return la.lu_solve((lu,piv),b)

    def update(x,y,µ,sigma = 0.1):

        #Keep track of how to update everything
        m,n = A.shape
        delta = solve(x,y,µ)
        deltaU = delta[n+m:]
        deltaU1 = deltaU[deltaU < 0]
        deltaY = delta[n:n+m]
        deltaY1 = deltaY[deltaY < 0]
        deltaX = delta[:n]
        deltaX1 = deltaX[deltaX < 0]
        minU = np.min(-µ[deltaU < 0]/deltaU1)
        bmax = np.min([1.,minU])
        minY = np.min(-y[deltaY < 0]/deltaY1)
        dmax = np.min([1.,minY])


        #final a, b, and d values we want
        b = np.min([1.,0.95*bmax])
        d = np.min([1.,0.95*dmax])
        a = min(b,d)

        x += a*deltaX
        y += a*deltaY
        µ += a*deltaU

        #return new values
        return x,y,µ

    x,y,µ = startingPoint(Q,c,A,b,guess)

    #iterate
    for i in range(niter):
        v = (1/m)*y.T@µ
        idk = np.hstack([np.zeros(n+m), .1*v*np.ones(m)])
        if la.norm(idk) < tol:
            break
        x,y,µ = update(x,y,µ)

    return x, c@x


def laplacian(n):
    """Construct the discrete Dirichlet energy matrix H for an n x n grid."""
    data = -1*np.ones((5, n**2))
    data[2,:] = 4
    data[1, n-1::n] = 0
    data[3, ::n] = 0
    diags = np.array([-n, -1, 0, 1, n])
    return spdiags(data, diags, n**2, n**2).toarray()


# Problem 3
def circus(n=15):
    """Solve the circus tent problem for grid size length 'n'.
    Display the resulting figure.
    """
    #Compute H, A, and c matrices
    H = laplacian(n)
    A = np.eye(n**2)
    c = -1/(n-1)**2 * np.ones(n**2)
    
    #Create the tent pole configuration
    L = np.zeros((n,n))
    L[n//2-1:n//2+1,n//2-1:n//2+1] = .5
    m = [n//6-1, n//6, int(5*(n/6.))-1, int(5*(n/6.))]
    mask1, mask2 = np.meshgrid(m, m)
    L[mask1, mask2] = .3
    L = L.ravel()
    
    #Set the initial guesses
    x = np.ones((n,n)).ravel()
    y = np.ones(n**2)
    mu = np.ones(n**2)
    
    #Calculate the solution using qInteriorPoint() function
    z = qInteriorPoint(H, c, A, L, (x,y,mu))[0].reshape((n,n))

    #plot
    domain = np.arange(n)
    X, Y = np.meshgrid(domain, domain)
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, z,  rstride=1, cstride=1, color='r')
    plt.title("Red Circus Tent")
    plt.show()


# Problem 4
def portfolio(filename="portfolio.txt"):
    """Markowitz Portfolio Optimization

    Parameters:
        filename (str): The name of the portfolio data file.

    Returns:
        (ndarray) The optimal portfolio with short selling.
        (ndarray) The optimal portfolio without short selling.
    """
    #Open the file and create the data
    with open(filename, 'r') as myfile:
        data = myfile.readlines()
    for i in range(len(data)):
        data[i]=data[i][:-1].split(' ')
    my_vals = np.zeros((22,8))
    
    #Create a matrix of all the values
    for i in range(22):
        for j in range(8):
            my_vals[i][j] = float(data[i][j+1])
    mus = np.mean(my_vals, axis=0)
    my_cov = 1.*np.cov(my_vals.T)
    P = 1/2*matrix(my_cov)
    A = np.ones((2,8))
    A[1] = mus
    
    #Allow short-selling at first
    A = matrix(A)
    q = matrix(1.*np.zeros(8))
    b = matrix(1.*np.array([1,1.13]))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P=P,q=q,A=A,b=b)
    short_sell = sol['x']
    
    #Disallow short-selling so all values are positive
    G = matrix(-1.*np.eye(8))
    h = matrix(1.*np.zeros(8))
    other_sol = solvers.qp(P=P,q=q,G=G,h=h,A=A,b=b)
    solvers.options['show_progress'] = False
    no_short = other_sol['x']
    return(np.array(short_sell), np.array(no_short))
    
