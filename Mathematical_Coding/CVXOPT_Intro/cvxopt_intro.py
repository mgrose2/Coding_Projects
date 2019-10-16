# cvxopt_intro.py
"""Volume 2: Intro to CVXOPT.
<Mark Rose>
<Section 2>
<3/13/19>
"""

from cvxopt import matrix, solvers
import numpy as np

def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x + 2y          >= 3
                    2x + 10y + 3z   >= 10
                    x               >= 0
                    y               >= 0
                    z               >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (sol['primal objective'])
    """
    #Make the numpy arrays
    c= np.array([2.,1.,3.])
    G = np.array([[-1.,-2.,0.],[-2.,-10.,-3.],[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]])
    h = np.array([-3.,-10.,0.,0.,0.])
    #Convert to CVXOPT matrix type
    c = matrix(c)
    G = matrix(G)
    h = matrix(h)
    #Use the predefined methods to solve the linear programming problem and return the answers.
    solvers.options['show_progress'] = False
    sol = solvers.lp(c,G,h)
    return(np.ravel(sol['x']), sol['primal objective'])
    raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray), without any slack variables u
        The optimal value (sol['primal objective'])
    """
    n = len(A[0,:])

    #Create G:
    G_1 = np.hstack((-np.eye(n),np.eye(n)))
    G_2 = np.hstack((-np.eye(n),-np.eye(n)))
    G_3 = np.hstack((-np.eye(n),np.zeros(n**2).reshape(n,n)))
    G = matrix(np.vstack((G_1,G_2)))
    #Create A:
    A = matrix(np.hstack((np.zeros_like(A),A*1.)))
    #Create c:
    c = matrix(np.hstack((np.ones(n),np.zeros(n))))
    #Create h:
    h = matrix(np.zeros(2*n))
    #Optimize
    solvers.options['show_progress'] = False
    sol = solvers.lp(c, G, h, A, matrix(b*1.))
    return np.ravel(sol['x'][n:]), sol['primal objective']
    raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (sol['primal objective'])
    """
    #Create all of the matrices
    c = matrix(np.array([4.,7.,6.,8.,8.,9.]))
    A = matrix(np.array([[1.,1.,0.,0.,0.,0.],
                         [0.,0.,1.,1.,0.,0.],
                         [0.,0.,0.,0.,1.,1.],
                         [1.,0.,1.,0.,1.,0.]]))
    b = matrix(np.array([7.,2.,4.,5.]))
    G = matrix(np.vstack((-1*np.eye(6), np.array([0.,1.,0.,1.,0.,1.]), -1*np.array([0.,1.,0.,1.,0.,1.]))))
    h = matrix(np.concatenate((np.zeros(6),np.array([8.,-8.]))))
    #Used the predefined functions to solve the function
    solvers.options['show_progress'] = False
    sol = solvers.lp(c, G, h, A, b)
    #Return the optimal value
    return np.ravel(sol['x']), sol['primal objective']
    raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (sol['primal objective'])
    """
    #Make the matrices
    Q = matrix(np.array([[3.,2.,1.],[2.,4.,2.],[1.,2.,3.]]))
    r = matrix(np.array([3.,0.,1.]))
    #Solve the problem
    sol = solvers.qp(Q,r)
    return np.ravel(sol['x']), sol['primal objective']
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def l2Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_2
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (sol['primal objective'])
    """
    #Create all of the matrices
    n = A.shape[1]
    Q = matrix(2*np.eye(n))
    r = matrix(np.zeros(n))
    A = matrix(A*1.)
    b = matrix(b*1.)
    #Solve the problems
    sol = solvers.qp(Q,r,A=A,b=b)
    return np.ravel(sol['x']), sol['primal objective']
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def prob6():
    """Solve the allocation model problem in 'ForestData.npy'.
    Note that the first three rows of the data correspond to the first
    analysis area, the second group of three rows correspond to the second
    analysis area, and so on.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (sol['primal objective']*-1000)
    """
    #Load the data
    forest_data = np.load('ForestData.npy')
    #Create all of the arrays
    p = -forest_data[:,3]
    t = -forest_data[:,4]
    g = -forest_data[:,5]
    w = -forest_data[:,6]
    #Create the matrices
    c = matrix(p)
    A = np.zeros((7,21))
    for k in range(7):
        A[k,3*k:3*(k+1)] = 1
    #Get all of the values and plug it into the solver
    b = np.array([75.,90.,140.,60.,212.,98.,113])

    G = matrix(np.vstack((A,-A,t,g,w/788,-np.eye(21))))
    h = matrix(np.hstack((b, -b, np.array([-40000.,-5.,-70.,]),np.zeros(21))))

    sol = solvers.lp(c, G, h)
    #Return the solver
    return np.ravel(sol['x']), -1000*sol['primal objective']
    raise NotImplementedError("Problem 6 Incomplete")
