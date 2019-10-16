# solutions.py
"""Volume 2: Gradient Descent Methods.
<Mark Rose>
<Section 2>
<2/20/19>
"""
import scipy.optimize as opt
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


# Problem 1
def steepest_descent(f, Df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the exact method of steepest descent.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #Set the alpha minimizer function
    af = lambda alpha: f(x0 - alpha*Df(x0).T)
    converged = False
    for i in range(maxiter):
        #Calculate each alpha and each xk
        ak = opt.minimize_scalar(af).x
        xk = x0 - ak*Df(x0).T
        if (la.norm(xk-x0) < tol):
            converged=True
            break
        x0=xk
    #Return the last value, whether or not it converged, and the number of iterations
    return(xk, converged, i+1)
    raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def conjugate_gradient(Q, b, x0, tol=1e-4):
    """Solve the linear system Qx = b with the conjugate gradient algorithm.

    Parameters:
        Q ((n,n) ndarray): A positive-definite square matrix.
        b ((n, ) ndarray): The right-hand side of the linear system.
        x0 ((n,) ndarray): An initial guess for the solution to Qx = b.
        tol (float): The convergence tolerance.

    Returns:
        ((n,) ndarray): The solution to the linear system Qx = b.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #Set all of the initial values
    n = len(b)
    r0 = Q.dot(x0) - b
    d0 = -1*r0
    k = 0
    rk = r0
    dk = d0
    xk = x0
    converged = False
    while(la.norm(rk) >=tol and k <=n):
        #Change each of the ak, rk, dk, and xk
        ak = (rk.T @ rk)/(dk.T @ Q @ dk )
        xk = xk + ak*dk
        rk1 = rk + ak* (Q @ dk)
        bk = (rk1.T @ rk1)/(rk.T @ rk)
        dk = -1*rk1 + bk* dk
        #Increase the value of k
        k = k+1
        rk = rk1
    #Check for convergence 
    if (la.norm(rk) < tol):
        converged = True
    return(xk, converged, k)
    raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def nonlinear_conjugate_gradient(f, df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the nonlinear conjugate gradient
    algorithm.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #Set all of the initial values for a, r, and x
    r0 = -1*df(x0).T
    d0 = r0
    xk= x0
    af = lambda alpha: f(xk + alpha*d0)
    a0 = opt.minimize_scalar(af).x
    xk = x0 + a0*d0
    x0 = xk
    rk = r0
    converged=False
    for i in range(maxiter):
        #Calcluate all of the new values for the a, r, and x
        rk = -1*df(xk).T
        Bk = (rk.T @ rk) / (r0.T @ r0)
        dk = rk + Bk*d0
        d0 = dk
        ak = opt.minimize_scalar(af).x
        r0 = rk
        xk = xk + ak*dk
        #Check to see if the value converged
        if (la.norm(xk-x0) < tol):
            converged=True
            break
        x0 = xk
    #Return the value, whether or not it converged, and the number of iterations
    return(xk, converged, i+1)
    raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def prob4(filename="linregression.txt",
          x0=np.array([-3482258, 15, 0, -2, -1, 0, 1829])):
    """Use conjugate_gradient() to solve the linear regression problem with
    the data from the given file, the given initial guess, and the default
    tolerance. Return the solution to the corresponding Normal Equations.
    """
    #Read in the data and define A and B
    data = np.loadtxt('linregression.txt')
    b = data[:,0]
    A = np.hstack((np.reshape(np.ones_like(b),(len(b),1)), data[:,1:]))
    #Use problem 2 to calculate the value that the least-squares problem converges to
    return(conjugate_gradient(A.T@A, A.T@b, x0)[0])
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
class LogisticRegression1D:
    """Binary logistic regression classifier for one-dimensional data."""

    def fit(self, x, y, guess):
        """Choose the optimal beta values by minimizing the negative log
        likelihood function, given data and outcome labels.

        Parameters:
            x ((n,) ndarray): An array of n predictor variables.
            y ((n,) ndarray): An array of n outcome variables.
            guess (array): Initial guess for beta.
        """
        #Define the negative log likelihood as a function
        f = lambda beta: np.sum(np.log(1+ np.exp(-1*(beta[0]+beta[1]*x)))+ (1-y)*(beta[0]+beta[1]*x))
        #Get an optimal beta value and set those as attributes
        B1 = opt.fmin_cg(f, guess)
        self.b0= B1[0]
        self.b1= B1[1]
        

    def predict(self, x):
        """Calculate the probability of an unlabeled predictor variable
        having an outcome of 1.

        Parameters:
            x (float): a predictor variable with an unknown label.
        """
        #Calculate and return the predicted value
        sig = 1/(1+np.exp(-(self.b0+self.b1*x)))
        return(sig)


# Problem 6
def prob6(filename="challenger.npy", guess=np.array([20., -1.])):
    """Return the probability of O-ring damage at 31 degrees Farenheit.
    Additionally, plot the logistic curve through the challenger data
    on the interval [30, 100].

    Parameters:
        filename (str): The file to perform logistic regression on.
                        Defaults to "challenger.npy"
        guess (array): The initial guess for beta.
                        Defaults to [20., -1.]
    """
    #Load the data from the file
    data = np.load('challenger.npy')
    x = data[:,0]
    y = data[:,1]
    #Instantiate the class, and fir the data
    my_log = LogisticRegression1D()
    my_log.fit(x,y,guess)
    dom = np.linspace(30,100,200)
    cod = []
    #Calculate predicted values to plot on the graph
    for i in dom:
        cod.append(my_log.predict(i))
    plt.plot(dom, np.array(cod), label='Logistic Fit')
    #Plot and scatter all of the values and label the axis
    plt.scatter(x,y, label='Previous Damage')
    plt.scatter(31, my_log.predict(31), label='P(Damage) at Launch')
    plt.xlabel('Temperature')
    plt.ylabel('O-Ring Damage')
    plt.title('Probability of O-Ring Damage')
    plt.legend()
    plt.show()
    return(my_log.predict(31))
    raise NotImplementedError("Problem 6 Incomplete")
