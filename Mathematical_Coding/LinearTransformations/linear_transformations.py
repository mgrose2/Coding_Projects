# linear_transformations.py
"""Volume 1: Linear Transformations.
<Name>
<Class>
<Date>
"""

from random import random
import numpy as np
from matplotlib import pyplot as plt
import time

# Problem 1
def stretch(A, a, b):
    """Scale the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    transf = np.array([[a,0],[0,b]])                                            #Multiply a matrix of a photo by a 2X2 matrix to stretch the photo
    return(transf.dot(A))
    raise NotImplementedError("Problem 1 Incomplete")

def shear(A, a, b):
    """Slant the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    transf = np.array([[1,a],[b,1]])                                            #Multiply a matrix of a photo by a 2X2 matrix to shear the photo
    return(transf.dot(A))
    raise NotImplementedError("Problem 1 Incomplete")

def reflect(A, a, b):
    """Reflect the points in A about the line that passes through the origin
    and the point (a,b).

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    """
    transf = 1/(a**2+b**2)*np.array([[a**2-b**2,2*a*b],[2*a*b,b**2-a**2]])      #Multiply a matrix of a photo by a 2X2 matrix to reflect the photo
    return(transf.dot(A))
    raise NotImplementedError("Problem 1 Incomplete")

def rotate(A, theta):
    """Rotate the points in A about the origin by theta radians.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    """
    transf = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])  #Multiply a matrix by this 2X2 Matrix to rotate it by a specific angle
    return(transf.dot(A))
    raise NotImplementedError("Problem 1 Incomplete")

def side_by_side(A, Ap):
    fig, axes= plt.subplots(1,2)
    axes[0].plot(A[0],A[1], 'k,')
    axes[0].axis([-1,1,-1,1])
    axes[0].set_aspect("equal")
    axes[1].plot(Ap[0],Ap[1], 'k,')                                                     #A function I made to compare the original horse picture to the modified picture in problem 1.
    axes[1].axis([-1,1,-1,1])
    axes[1].set_aspect("equal")
    plt.show()
    return


# Problem 2
def solar_system(T, x_e, x_m, omega_e, omega_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (x_e,0) and the initial
    position of the moon is (x_m,0).

    Parameters:
        T (int): The final time.
        x_e (float): The earth's initial x coordinate.
        x_m (float): The moon's initial x coordinate.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """
    e_final = np.zeros((2,1000))
    m_final = np.zeros((2,1000))
    time_vec = np.linspace(0,T, 1000)
    y_e = 0
    y_m = 0
    earth_in = np.array([x_e, y_e])
    moon_in = np.array([x_m, y_m])                                                          #Initialize the moon and the earths original positions
    for i in range(1000):
        e_final[:,i] = rotate(earth_in, time_vec[i]*omega_e)                                #Rotate the Moon and the Earth according to the angles and omegas that were given
        m_final[:,i] = rotate(moon_in-earth_in, time_vec[i]*omega_m)
    m_final = m_final+e_final                                                               
    plt.plot(e_final[0], e_final[1], 'b-',linewidth=3.0, label='Earth')
    plt.plot(m_final[0], m_final[1], 'r-',linewidth=3.0, label='Moon')
    plt.axis([-12,12,-12,12])
    plt.gca().set_aspect("equal")
    plt.legend()
    plt.show()
    return
    raise NotImplementedError("Problem 2 Incomplete")


def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]

# Problem 3
def prob3():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """
    n_list = []
    for i in range(1,9):
        n_list.append(2**i)
    mv_times = []
    mm_times = []
    
    for n in n_list:
        A = random_matrix(n)
        B = random_matrix(n)
        x = random_vector(n)
        start_mv = time.time()
        matrix_vector_product(A,x)                                                          #time the matrix matrix and matrix vector product times 
        time_mv = time.time()-start_mv
        mv_times.append(time_mv)
        start_mm = time.time()
        matrix_matrix_product(A,B)
        time_mm = time.time()-start_mm
        mm_times.append(time_mm)
    
    
    fig, axes= plt.subplots(1,2)
    axes[0].plot(n_list, mv_times, 'o-')                                                    #graph the times against each other
    axes[0].set_title('Matrix Vector Multiplication')
    axes[0].set_xlabel('n')
    axes[0].set_ylabel('Seconds')
    axes[1].plot(n_list, mm_times, 'o-')
    axes[1].set_title('Matrx Matrix Multiplication')
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('Seconds')
    
    plt.show()
    return
        
    
    raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def prob4():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot()
    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    n_list = []
    for i in range(1,9):
        n_list.append(2**i)
    mv_times = []
    mm_times = []
    vd_times = []
    md_times = []
    
    for n in n_list:
        A = random_matrix(n)
        B = random_matrix(n)
        x = random_vector(n)                                                                        
        start_mv = time.time()
        matrix_vector_product(A,x)
        time_mv = time.time()-start_mv
        mv_times.append(time_mv)
        start_mm = time.time()                                                              #Time the matrix matrix, matrix vector, dot matrix, and dot vector functions             
        matrix_matrix_product(A,B)
        time_mm = time.time()-start_mm
        mm_times.append(time_mm)
        A = np.array(A)
        B = np.array(B)
        x = np.array(x)
        start_vd = time.time()
        A@x
        time_vd = time.time()-start_vd
        vd_times.append(time_vd)
        start_md = time.time()
        A@B
        time_md = time.time()-start_md
        md_times.append(time_md)
        
    fig, axes= plt.subplots(1,2)
    axes[0].plot(n_list, mv_times,'b-o', label = 'Matrix-Vector Times')
    axes[0].plot(n_list, mm_times,'g-o', label = 'Matrix-Matrix Times')                    #Graph all of the times on one axes
    axes[0].plot(n_list, vd_times,'r-o', label = 'Vector-Dot Times')
    axes[0].plot(n_list, md_times,'c-o', label = 'Vector-Matrix Times')
    axes[0].legend()
    axes[0].set_title('Multiplication Execution Times')
    axes[0].set_xlabel('n')
    axes[0].set_ylabel('Seconds')
    axes[1].loglog(n_list, mv_times,'b-o', label = 'Matrix-Vector Times')
    axes[1].loglog(n_list, mm_times,'g-o', label = 'Matrix-Matrix Times')                   #Graph all of the log times on another axes
    axes[1].loglog(n_list, vd_times,'r-o', label = 'Vector-Dot Times')
    axes[1].loglog(n_list, md_times,'c-o', label = 'Vector-Matrix Times')
    axes[1].legend()
    axes[1].set_title('Log Multiplication Execution Time')
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('Seconds')
    
    plt.show()
    return

    
    raise NotImplementedError("Problem 4 Incomplete")

