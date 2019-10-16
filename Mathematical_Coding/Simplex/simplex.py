# simplex.py
"""Volume 2: Simplex.
<Mark Rose>
<Section 2>
<3/7/19>


Problems 1-6 give instructions on how to build the SimplexSolver class.
The grader will test your class by solving various linear optimization
problems and will only call the constructor and the solve() methods directly.
Write good docstrings for each of your class methods and comment your code.

prob7() will also be tested directly.
"""

import numpy as np

# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        maximize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    def __init__(self, c, A, b):
        """Check for feasibility and initialize the tableau.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        #Get the shape of A and save the dimensions as the amount of basic and nonbasic variables
        m,n = np.shape(A)
        self.bas_size = m
        self.nonbas_size = n
        #Raise a value error if the problem is not feasible at the origin.
        if np.any(b<0):
            raise ValueError('The Problem is Not Feasible at the Origin')
        self.c = c
        self.A = A
        self.b = b
        #After setting the variables as attributes, create the Tableau
        self.var = np.hstack((np.arange(n,n+m), np.arange(n)))
        c_hat = np.hstack((c, np.zeros(m)))
        A_hat = np.hstack((A, np.eye(m)))
        T_top = np.hstack((0, -1*c_hat.T, 1))
        T_bot = np.hstack((np.reshape(b, (len(b),1)), A_hat, np.reshape(np.zeros(m), (m,1))))
        #Save the tableau and its dimensions as attributes within the class.
        self.tableau = np.vstack((T_top, T_bot))
        m,n = np.shape(self.tableau)
        self.m = m
        self.n = n
        return 
        NotImplementedError("SimplexSolver.__init__() Incomplete")

    def pivot_find(self):
        """Function for finding the column and row that the pivot should happen. 
        
        Raises: 
            Value Error if the problem is unbounded.
            
        Returns:
            (int): The column to pivot at.
            (int): The row to pivot at.
        """
        #Instantiate various variables to be used in the function
        pivot_col = 0
        pivot_row = 1
        min_val = 0
        counter = []
        #Find the column to pivot at by finding the first negative value.
        for i in range(self.n-1):
            if self.tableau[0,i+1] < 0:
                pivot_col = i+1
                break
        #Raise a value error if all the values in the pivot column are non positive.
        if np.all(self.tableau[1:,pivot_col] <= 0)==True:
            raise ValueError("Unbounded Optimization Problem")
        #Find the ratio of each of the variables to find the pivot row
        for j in range(self.m-1):
            if self.tableau[j+1, pivot_col]>0:
                ratio = self.tableau[j+1,0]/self.tableau[j+1, pivot_col]
                if min_val!=0:
                    #If the ratio is the smallest value make it the new pivot row
                    if (ratio < min_val):
                        min_val = ratio
                        pivot_row = j+1
                        counter.clear()
                        counter.append(pivot_row)
                    #If the ratio is the same as the minimal value, append it to the list
                    elif (ratio == min_val):
                        counter.append(j+1)
                else:
                    #This is the first ratio value and we should append it to initialize the pivot row
                    min_val = ratio
                    pivot_row = j+1
                    counter.append(pivot_row)
        #Find the pivot row by taking the arg min of the tableau optimization line
        pivot_row= counter[np.argmin(self.tableau[counter,pivot_col])]
        return(pivot_col,pivot_row)
    
    def pivot_op(self):
        """ A function which does the actual pivot operation.
        """
        #Find the pivot row and pivot column using the function above
        pivot_col, pivot_row = self.pivot_find()
        enter = pivot_col-1
        exit = self.var[pivot_row-1]
        #Define the entering and leaving variables
        enter_ind = np.where(self.var==enter)
        exit_ind = np.where(self.var==exit)
        #Change my variables list after swtiching the variables
        self.var[enter_ind],self.var[exit_ind]=self.var[exit_ind],self.var[enter_ind]
        self.tableau[pivot_row]/=self.tableau[pivot_row,pivot_col]
        #Update the Tableau
        for i in range(self.m):
            if (i != pivot_row):
                ind_val = self.tableau[i, pivot_col]
                self.tableau[i]+= (-1*ind_val*self.tableau[pivot_row])
        return
            
        
        
    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The maximum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        #Keep optimizing while the objective function still has a negative value.
        while np.any(self.tableau[0,1:] <0):
            self.pivot_op()
        basic_dict = {}
        non_basic_dict = {}
        #Get the basic values and put them into a dictionary
        for i in range(self.bas_size):
            basic_dict[self.var[i]] = self.tableau[i+1,0] 
        #Get the nonbasic values and put them also into a dictionary
        for j in range(self.nonbas_size):
            non_basic_dict[self.var[j+self.bas_size]] = 0
        #Return the answer
        return(self.tableau[0,0], dict(sorted(basic_dict.items())),dict(sorted(non_basic_dict.items())))
        
        raise NotImplementedError("SimplexSolver.solve() Incomplete")


# Problem 7
def prob7(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        The minimizer of the problem (as an array).
    """
    #Load the data and assign them variable names
    data = np.load(filename)
    a = data['A']
    p = data['p']
    m = data['m']
    d = data['d']
    #Create my c, A, and b matrices
    c = p
    A = np.vstack((a, np.eye(len(d))))
    b = np.hstack((m,d))
    my_simp = SimplexSolver(c,A,b)
    #Solve the optimization problem using the simplex method
    ans_dict = my_simp.solve()[1]
    my_ans = np.zeros(len(p))
    #Put my answer into a vector and return it
    for i in range(len(p)):
        my_ans[i] = ans_dict[i]
    return(my_ans)
    raise NotImplementedError("Problem 7 Incomplete")
