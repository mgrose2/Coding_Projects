# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
<Mark Rose>
<Math 345>
<9/5/18>
"""
import numpy as np
import matplotlib.pyplot as plt

# Problem 1
def var_of_means(n):
    """Construct a random matrix A with values drawn from the standard normal
    distribution. Calculate the mean value of each row, then calculate the
    variance of these means. Return the variance.

    Parameters:
        n (int): The number of rows and columns in the matrix A.

    Returns:
        (float) The variance of the means of each row.
    """
    A = np.random.normal(size=(n,n))
    mean_mat = np.mean(A, axis=1)
    return(np.var(mean_mat))
    raise NotImplementedError("Problem 1 Incomplete")

def prob1():
    """Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """
    A = np.zeros(10)
    for i in range(100,1100,(100)):
        j=int((i/100-1))
        A[j]=var_of_means(i)
    plt.plot(A)
    plt.show()
    return
    raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def prob2():
    """Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    y = np.cos(x)
    z = np.sin(x)
    a = np.arctan(x)
    plt.plot(x,y,)
    plt.plot(x,z,)
    plt.plot(x,a)
    plt.show()
    return
    raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def prob3():
    """Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Set the range of the x-axis to [-2,6] and the range of the
           y-axis to [-6,6].
    """
    x = np.linspace(-2,1,100)
    xx = np.linspace(1,6,100)
    x=np.delete(x,99)
    xx=np.delete(xx,0)
    y = 1/(x-1)
    yy = 1/(xx-1)
    plt.plot(x, y, 'm--', linewidth=4)
    plt.plot(xx, yy, 'm--',linewidth=4)
    plt.xlim(-2,6)
    plt.ylim(-6,6)
    plt.show()
    return
    raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def prob4():
    """Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi].
        1. Arrange the plots in a square grid of four subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """
    x = np.linspace(0, 2*np.pi, 100)
    fig, axes= plt.subplots(2,2)
    axes[0,0].plot(x,np.sin(x), 'g')
    axes[0,1].plot(x,np.sin(2*x), 'r--')
    axes[1,0].plot(x,2*np.sin(x), 'b--')
    axes[1,1].plot(x,2*np.sin(2*x), 'm:')
    axes[0,0].set_title('sin(x) Graph')
    axes[0,1].set_title('sin(2x) Graph')
    axes[1,0].set_title('2sin(x) Graph')
    axes[1,1].set_title('2sin(2x) Graph')
    axes[0,0].axis([0,2*np.pi,-2,2])
    axes[0,1].axis([0,2*np.pi,-2,2])
    axes[1,0].axis([0,2*np.pi,-2,2])
    axes[1,1].axis([0,2*np.pi,-2,2])
    fig.suptitle('Sin Experimentation')
    plt.show()
    return
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def prob5():
    """Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes against latitudes. Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """
    data = np.load('FARS.npy')
    fig, axes= plt.subplots(1,2)
    axes[0].plot(data[:,1],data[:,2],'k,')
    axes[0].set_aspect("equal")
    axes[0].set_title('U.S. Accident Map')
    axes[0].set_xlabel('Latitude') 
    axes[0].set_ylabel('Longitude')  
    axes[1].hist(data[:,0], bins=np.arange(0,25)) #spacing?
    axes[1].set_xticks(range(24))
    axes[1].set_xlim(0,24)
    axes[1].set_xlabel('Time When Accidents Occur') 
    axes[1].set_ylabel('Frequency')
    
    plt.show()
    return
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def prob6():
    """Plot the function f(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of f, and one with a contour
            map of f. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Add a colorbar to each subplot.
    """
    fig, axes = plt.subplots(1,2)
    x = np.linspace(-2*np.pi,2*np.pi,100)
    y = np.linspace(-2*np.pi,2*np.pi,100)
    X,Y = np.meshgrid(x,y)
    Z = np.sin(X)*np.sin(Y)/(X*Y)
    axes[0].pcolormesh(X, Y, Z, cmap="viridis")
    plt.colorbar(axes[0].pcolormesh(X, Y, Z, cmap="viridis"), ax=axes[0])
    axes[0].set_xlim(-2*np.pi, 2*np.pi)
    axes[0].set_ylim(-2*np.pi, 2*np.pi)
    axes[1].contourf(X,Y,Z, 10, cmap='coolwarm')
    plt.colorbar(axes[1].contourf(X,Y,Z, 10, cmap='coolwarm'))
    axes[1].set_xlim(-2*np.pi, 2*np.pi)
    axes[1].set_ylim(-2*np.pi, 2*np.pi)
    plt.show()
    return
    raise NotImplementedError("Problem 6 Incomplete")


if __name__ == '__main__':
    prob6()
