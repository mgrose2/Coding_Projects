# dynamic_programming.py
"""Volume 2: Dynamic Programming.
<Mark Rose>
<Section 2>
<4/10/19>
"""

import numpy as np
import matplotlib.pyplot as plt


def calc_stopping(N):
    """Calculate the optimal stopping time and expected value for the
    marriage problem.

    Parameters:
        N (int): The number of candidates.

    Returns:
        (float): The maximum expected value of choosing the best candidate.
        (int): The index of the maximum expected value.
    """
    #Create an array of expected values by using the equation from 17.1
    exp_val = np.zeros(N)
    for i in range(N-1):
        exp_val[N-i-2] = (N-i-1)/(N-i)*exp_val[N-i-1] + 1/N
    #Change the first value so that it is 1/N
    exp_val[0] = 1/N
    return(max(exp_val), np.argmax(exp_val))
    raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def graph_stopping_times(N):
    """Graph the optimal percentage of candidates to date optimal
    and expected value for the marriage problem for n=3,4,...,M.

    Parameters:
        M (int): The maximum number of candidates.

    Returns:
        (float): The optimal stopping percent of candidates for N.
    """
    #Instantiate variables
    stop_per = []
    max_prob = []
    my_vals = np.arange(3,N+1)
    #Use the previous problem to get the probability and stopping index
    for N in my_vals:
        prob, ind = calc_stopping(N)
        stop_per.append(ind/N)
        max_prob.append(prob)
    #Plot all of the values
    plt.plot(my_vals, stop_per, label = "Stopping Percentage")
    plt.plot(my_vals, max_prob, label = "Maximum Probability")
    plt.xlabel('N Values')
    plt.title("Marriage Problem")
    plt.legend()
    plt.show()
    #Return the stopping percentate for N
    return(stop_per[-1])
    
    raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def get_consumption(N, u=np.sqrt):
    """Create the consumption matrix for the given parameters.

    Parameters:
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        u (function): Utility function.

    Returns:
        ((N+1,N+1) ndarray): The consumption matrix.
    """
    #Create the W values
    w = np.linspace(0,1,N+1)
    C = np.zeros((N+1,N+1))
    #Update the matrix
    for i in range(N+1):
        for j in range(N+1):
            if w[i]-w[j] > 0:
                C[i][j] = u(w[i]-w[j])
    return(C)
    raise NotImplementedError("Problem 3 Incomplete")


# Problems 4-6
def eat_cake(T, N, B, u=np.sqrt):
    """Create the value and policy matrices for the given parameters.

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        A ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            value of having w_i cake at time j.
        P ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            number of pieces to consume given i pieces at time j.
    """
    #Get w, A, and P
    w = np.linspace(0,1,N+1)
    A = np.zeros((N+1, T+1))
    P = np.zeros((N+1,T+1))
    #Update the last column of the A and P matrices
    for i in range(N+1):
        A[i,T] = u(w[i])
        P[i,T] = w[i]
    my_cv = []
    CV = np.zeros((N+1,N+1))
    #Create the CV matrix
    for l in range(1,T+1):
        for i in range(N+1):
            for j in range(N+1):
                #Change the value only if it is greater than or equal to zero
                if (w[i]-w[j] >= 0):
                    CV[i,j] = u(w[i]-w[j]) + B*A[j][-1*l]
        #Update the A matrix
        A[:,-1*l-1] = np.max(CV, axis=1) 
        my_cv.append(CV.copy())
    my_cv.reverse()
    #Update the P Matrix
    for i in range(N+1):
        for j in range(T):
            CV = my_cv[j]
            my_ind = np.argmax(CV[i,:-1])
            P[i,j] = w[i] - w[my_ind]
    return(A,P)
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 7
def find_policy(T, N, B, u=np.sqrt):
    """Find the most optimal route to take assuming that we start with all of
    the pieces. Show a graph of the optimal policy using graph_policy().

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        ((N,) ndarray): The matrix describing the optimal percentage to
            consume at each time.
    """
    #Get the Matrices
    A,P = eat_cake(T,N,B,u)
    my_pol = np.zeros(T+1)
    num = N
    #Update the policy
    for i in range(T+1):
        my_pol[i] = P[num,i]
        num = num-int(N*P[num,i])
    return(np.max(A), my_pol)
    raise NotImplementedError("Problem 7 Incomplete")
