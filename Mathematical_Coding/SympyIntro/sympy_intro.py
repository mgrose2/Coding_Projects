# sympy_intro.py
"""Python Essentials: Introduction to SymPy.
<Mark Rose>
<Section 2>
<1/15/19>
"""
import sympy as sy
import matplotlib.pyplot as plt
import numpy as np


# Problem 1
def prob1():
    """Return an expression for

        (2/5)e^(x^2 - y)cosh(x+y) + (3/7)log(xy + 1).

    Make sure that the fractions remain symbolic.
    """
    x,y = sy.symbols('x,y')                          #Define the variables
    return(sy.Rational(2,5)*sy.exp(x**2-y)*sy.cosh(x+y)+sy.Rational(3,7)*sy.log(x*y+1)) #Return the function
    raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def prob2():
    """Compute and simplify the following expression.

        product_(i=1 to 5)[ sum_(j=i to 5)[j(sin(x) + cos(x))] ]
    """
    x,i,j = sy.symbols('x i j')                      #Define the variables
    return(sy.simplify(sy.product(sy.summation(j*(sy.sin(x) + sy.cos(x)), (j,i,5)), (i,1,5))))  #Use summation, then product, then simplify
    raise NotImplementedError("Problem 2 Incomplete")



# Problem 3
def prob3(N):
    """Define an expression for the Maclaurin series of e^x up to order N.
    Substitute in -y^2 for x to get a truncated Maclaurin series of e^(-y^2).
    Lambdify the resulting expression and plot the series on the domain
    y in [-3,3]. Plot e^(-y^2) over the same domain for comparison.
    """
    x, i, n, y = sy.symbols('x i n y')                                          #Define the variables
    expr = sy.summation(x**n/sy.factorial(n),(n,0,N))                           #Compute the summation
    expr = expr.subs(x, -y**2)                                                  #substitute the variables
    f = sy.lambdify(y, expr, 'numpy')                                           #Make it into a funciton
    dom = np.linspace(-2,2,100)
    plt.plot(dom, f(dom), label = 'Series Approximation')
    plt.plot(dom, np.exp(-1*dom**2), label='Actual Function')                   #Plot the original and a series for some N
    plt.legend()
    plt.show()
    return
    raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def prob4():
    """The following equation represents a rose curve in cartesian coordinates.

    0 = 1 - [(x^2 + y^2)^(7/2) + 18x^5 y - 60x^3 y^3 + 18x y^5] / (x^2 + y^2)^3

    Construct an expression for the nonzero side of the equation and convert
    it to polar coordinates. Simplify the result, then solve it for r.
    Lambdify a solution and use it to plot x against y for theta in [0, 2pi].
    """
    x, y, r, th = sy.symbols('x y r th')                                        #Define variables
    expr = 1- ((x**2+y**2)**(sy.Rational(7,2))+18*x**5*y-60*x**3*y**3+18*x*y**5)/(x**2+y**2)**3   #Make function
    expr = expr.subs({x:r*sy.cos(th), y:r*sy.sin(th)})                         #Replace with polar cooardinates
    expr = sy.simplify(expr)
    r1, r2 = sy.solve(expr,r)                                                  #Lambdify first return value
    r1 = sy.lambdify(th, r1, 'numpy')
    dom = np.linspace(0,2*np.pi, 100) 
    plt.plot(dom, r1(dom)*np.cos(dom), label='x Graph')                        #Plot x(t)=r(t)*cos(t) and y(t)= r(t)*sin(t) 
    plt.plot(dom, r1(dom)*np.sin(dom), label='y Graph')
    plt.legend()
    plt.show()
    return()
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def prob5():
    """Calculate the eigenvalues and eigenvectors of the following matrix.

            [x-y,   x,   0]
        A = [  x, x-y,   x]
            [  0,   x, x-y]

    Returns:
        (dict): a dictionary mapping eigenvalues (as expressions) to the
            corresponding eigenvectors (as SymPy matrices).
    """
    x, y, lam = sy.symbols('x y lam')
    A = sy.Matrix([[x-y-lam, x, 0],[x, x-y-lam, x],[0, x, x-y-lam]])           #Define the Matrix and variables
    e_vals = sy.solve(sy.det(A), lam)                                          #Solve for the determinant
    my_dic = {} 
    for i in e_vals:
        B = A.subs(lam, i)                                                     
        my_dic[i] = (B.nullspace()[0])                                         #Take the nullspace of the eigenvalues substituted in for lambda
    return(my_dic)                                                             #Return the eigenvalues corresponding to the eigenvectors
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def prob6():
    """Consider the following polynomial.

        p(x) = 2*x^6 - 51*x^4 + 48*x^3 + 312*x^2 - 576*x - 100

    Plot the polynomial and its critical points. Determine which points are
    maxima and which are minima.

    Returns:
        (set): the local minima.
        (set): the local maxima.
    """
    my_min = set()                                                             #Create a min and max set
    my_max = set()
    x = sy.symbols('x')                                                        #Define the variables
    f = 2*x**6-51*x**4+48*x**3+312*x**2-576*x-100
    crit = sy.solve(sy.diff(f,x), x)                                           #Solve for the critical points by setting the deriavtive equal to zero
    f2 = sy.lambdify(x,sy.diff(sy.diff(f,x),x), 'numpy')                       #Create a function for the scond derivative of p
    for i in crit:
        if f2(i) < 0:
            my_max.add(i)                                                      #Add into the max set if the second derivative is less than zero, and into the min set if it is greater than zero
        else:
            my_min.add(i)
    dom = np.linspace(-5,5,100)
    f = sy.lambdify(x, f, 'numpy')                                            #Create a callable function for p
    plt.plot(dom, f(dom), label = 'Polynomial', color = 'b')
    plt.scatter(np.array(list(my_max)), f(np.array(list(my_max))), label = 'Local Max', color = 'r') #Plot the maxs and the mins according to color
    plt.scatter(np.array(list(my_min)), f(np.array(list(my_min))), label = 'Local Min', color = 'g')
    plt.legend()
    plt.show()
    return(my_min, my_max)
    raise NotImplementedError("Problem 6 Incomplete")


# Problem 7
def prob7():
    """Calculate the integral of f(x,y,z) = (x^2 + y^2 + z^2)^2 over the
    sphere of radius r. Lambdify the resulting expression and plot the integral
    value for r in [0,3]. Return the value of the integral when r = 2.

    Returns:
        (float): the integral of f over the sphere of radius 2.
    """
    x, y, z, p, fi, t = sy.symbols('x y z p fi t')                                #Define the variables
    expr = (x**2+y**2+z**2)**2                                                    
    g = sy.Matrix([p*sy.sin(fi)*sy.cos(t),p*sy.sin(fi)*sy.sin(t),p*sy.cos(fi)])   #Create a matrix for the variables
    J = g.jacobian([p, t, fi])                                                    #Get the jacobian of that matrix
    expr = expr.subs({x:p*sy.sin(fi)*sy.cos(t), y:p*sy.sin(fi)*sy.sin(t), z:p*sy.cos(fi)}) #Replace the variables with the polar coordinates
    expr = sy.simplify(expr *-1*J.det())
    f = sy.lambdify(p, sy.integrate(expr, (fi, 0, sy.pi), (t, 0, 2*sy.pi),p), 'numpy')  #Createa callable integral function
    dom = np.linspace(0,3,100)
    plt.plot(dom, f(dom))                                                         #Plot the integral
    plt.title('Volume of a Sphere')
    plt.xlabel('Radius')
    plt.ylabel('Volume')
    plt.show() 
    return(f(2))                                                                  #Return the value when the radius is 2
    raise NotImplementedError("Problem 7 Incomplete")

