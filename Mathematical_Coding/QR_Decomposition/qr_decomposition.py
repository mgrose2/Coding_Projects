# qr_decomposition.py
"""Volume 1: The QR Decomposition.
<Mark Rose>
<Section 2>
<10/29/18>
"""

import numpy as np
from scipy import linalg as la

# Problem 1
def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    m,n = np.shape(A)                                                                               #store the dimensions of A
    Q = A.copy()
    R = np.zeros((n,n))                                                                             #Initialize each matrix we want
    for i in range(n):
        R[i][i] = la.norm(Q[:,i])
        Q[:,i] = Q[:,i]/R[i][i]                                                                     #normalize the ith column
        for j in range(i+1, n):                                                                     #Calculate the decomposition using the algorithm given
            R[i][j] = Q[:,j].T.dot(Q[:,i])
            Q[:,j] = Q[:,j] - R[i][j]*(Q[:,i])                                                      #orthogonalize the jth column
    return Q,R
        
    raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    Q, R = qr_gram_schmidt(A)                                                                       #Get the Q, R from our previous function
    return(abs(np.prod(np.diag(R))))                                                                #Get the determinant by taking the product of the diagonal of R
    raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    m, n = np.shape(A)
    Q,R = qr_gram_schmidt(A)
    x = np.zeros(n)
    y = Q.T.dot(b)                                                                                   #Find y by taking the transpose of Q times b
    for i in range(len(A)-1,-1,-1):
        if i==len(A)-1:
            x[i] = 1/R[i,i]*y[i]                                                                     #Then use the y found in the previous result to solve Rx=y by back substitution
        else:
            x[i] = 1/R[i,i]*(y[i] - np.sum(np.dot(R[i,i+1:],x[i+1:])))
    return x     
    
    #Back substitution
    raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    sign = lambda x: 1 if x >=0 else -1
    m,n = np.shape(A)
    R = A.copy()
    Q = np.eye(m)                                                                       #the mxm identity
    for k in range(n):
        u = R[k:,k].copy()
        u[0] = u[0] + sign(u[0])*la.norm(u)                                             #change the first value in u
        u = u/la.norm(u) 
        R[k:,k:] = R[k:,k:] - 2*np.outer(u,u.T.dot(R[k:,k:]))                           #apply the reflection to R                              
        Q[k:,:] = Q[k:,:] - 2*np.outer(u,u.T.dot(Q[k:,:]))                              #Apply the reflection to Q
    return Q.T, R
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    sign = lambda x: 1 if x >=0 else -1                                                    #A way to use the sign even if 0 is involved
    m,n = np.shape(A)
    H = A.copy()
    Q = np.eye(m)
    for k in range(n-2):
        u = H[k+1:,k].copy()
        u[0] = u[0] + sign(u[0])*la.norm(u)
        u = u/la.norm(u)
        H[k+1:,k:] = H[k+1:,k:] - 2*np.outer(u,(u.T.dot(H[k+1:,k:])))                       #apply the Qk to H
        H[:,k+1:] = H[:,k+1:] - 2*np.outer((H[:,k+1:].dot(u)),(u.T))                        #Apply the Qk.T to H                                
        Q[k+1:,:] = Q[k+1:,:] - 2*np.outer(u,(u.T.dot(Q[k+1:,:])))                          #Apply the Qk to Q
    return H, Q.T
    raise NotImplementedError("Problem 5 Incomplete")
