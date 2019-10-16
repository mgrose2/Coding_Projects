# numpy_intro.py
"""Python Essentials: Intro to NumPy.
<Mark Rose>
<Math 345>
<9/5/18>
"""
import numpy as np

def prob1():
    """Define the matrices A and B as arrays. Return the matrix product AB."""
    A = np.array([[3,-1,4],[1,5,-9]])
    B = np.array([[2,6,-5,3],[5,-8,9,7],[9,-3,-2,-3]])
    return(A.dot(B))
    raise NotImplementedError("Problem 1 Incomplete")


def prob2():
    """Define the matrix A as an array. Return the matrix -A^3 + 9A^2 - 15A."""
    A = np.array([[3,1,4],[1,5,9],[-5,3,1]])
    return(-(A.dot(A.dot(A)))+9*A.dot(A)-15*A)
    raise NotImplementedError("Problem 2 Incomplete")


def prob3():
    """Define the matrices A and B as arrays. Calculate the matrix product ABA,
    change its data type to np.int64, and return it.
    """
    A = np.triu(np.ones((7,7)))
    B = np.tril(np.full((7,7),-6))+5
    return(A.dot(B.dot(A)).astype(np.int64))
    raise NotImplementedError("Problem 3 Incomplete")


def prob4(A):
    """Make a copy of 'A' and set all negative entries of the copy to 0.
    Return the copy.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    c = A.copy()
    b = A < 0
    c[b] = 0
    return(c)
    raise NotImplementedError("Problem 4 Incomplete")


def prob5():
    """Define the matrices A, B, and C as arrays. Return the block matrix
                                | 0 A^T I |
                                | A  0  0 |,
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    A = np.array([[0,2,4],[1,3,5]])
    B = np.tril(np.full((3,3),3))
    C = np.diag([-2,-2,-2])
    up_row = np.hstack((np.zeros((3,3)),A.T,np.eye(3)))
    mid_row = np.hstack((A, np.zeros((2,5))))
    bot_row = np.hstack((B, np.zeros((3,2)),C))
    combo = np.vstack((up_row, mid_row, bot_row))
    return(combo)
    raise NotImplementedError("Problem 5 Incomplete")


def prob6(A):
    """Divide each row of 'A' by the row sum and return the resulting array.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    return(A/A.sum(axis=1, keepdims=True))
    raise NotImplementedError("Problem 6 Incomplete")


def prob7():
    """Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid.
    """
    grid = np.load("grid.npy")
    horiz_max = np.max(grid[:,:-3] * grid[:,1:-2] * grid[:,2:-1] * grid[:,3:])
    vert_max = np.max(grid.T[:,:-3] * grid.T[:,1:-2] * grid.T[:,2:-1] * grid.T[:,3:])
    left_diag = np.max(grid[:-3,:-3] * grid[1:-2,1:-2]*grid[2:-1,2:-1]*grid[3:,3:])
    right_diag = np.max(grid[:-3,3:]*grid[1:-2,2:-1]*grid[2:-1,1:-2]*grid[3:,:-3])
    return(np.max(np.array([horiz_max, vert_max, left_diag, right_diag])))
    raise NotImplementedError("Problem 7 Incomplete")


if __name__ == '__main__':
    print(prob5())
