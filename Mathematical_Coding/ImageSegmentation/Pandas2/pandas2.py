# # Pandas 2
# 
# ## Mark Rose
# 
# ## Section 2
# 
# ## 10/8/19


import numpy as np
import pandas as pd
#from google.colab import files
import matplotlib.pyplot as plt


# Files needed
#      crime_data.csv
#      college.csv
#uploaded = files.upload()


def prob1():
    """
    Create 3 visualizations of the crime data set.
    One of the visualizations should be a histogram.
    The visualizations should be clearly labelled and easy to understand.
    """
    #Get the data
    df = pd.read_csv('crime_data.csv')
    new_df = df.set_index('Year')
    
    #Drop the Population and year and plot
    new_df.plot(y=['Violent', 'Property', 'Murder'],linewidth=1)
    plt.ylabel('Amount of Murders')
    plt.title('Series of Crimes per Year')
    
    
    #Plot the data
    df.plot(kind='hist', y=['Larceny', 'Property', 'Forcible Rape'],stacked=True)
    plt.title('Crimes by Amount')
    plt.ylabel('Amount of Years')
    plt.xlabel('Number of Crime Instances')
    plt.show()
    return
    raise NotImplementedError("Problem 1 Incomplete")    



def prob2():
       """
       Use visualizations to identify if trends exists between Forcible Rape
       and
           1. Violent
           2. Burglary
           3. Aggravated Assault
       Plot each visualization.
       Return a tuple of booleans. Each entry should identify whether that 
       element correlates or not.
       """
       #Get the data and see if there is a correlation between rape and different crimes
       df = pd.read_csv('crime_data.csv')
       df.plot(kind='scatter', x='Forcible Rape', y='Violent')
       plt.title("Violence vs. Forcible Rape")
       df.plot(kind='scatter', x='Forcible Rape', y='Burglary')
       plt.title("Burglary vs. Forcible Rape")
       df.plot(kind='scatter', x='Forcible Rape', y='Aggravated Assault')
       plt.title("Aggravated Assault vs. Forcible Rape")
       plt.show()
       #Return the answers
       return(True, False, True)



def prob3():
    """
    Use crime_data.csv to display the following distributions.
        1. The distributions of Burglary, Violent, and Vehicle Theft across all crimes
        2. The distributions of Vehicle Thefts by the number of Robberies
    """
    #Get the crime data and plot them
    df = pd.read_csv('crime_data.csv')
    df.plot(kind="box", y=['Burglary', 'Violent', 'Vehicle Theft'])
    plt.ylabel('Crime Frequency')
    plt.title('Distributions of Crimes')
    df.plot(kind="Hexbin", x='Vehicle Theft', y='Robbery', gridsize=15)
    plt.title('Distributions of Vehicle Thefts by Robberies')
    plt.show()
    return
    


def prob4():
    """
    Answer the following questions with the College dataset
    
    1. Hexbin between top10perc and PhD
    2. Looking at applications, acceptance, and enrollment
    3. Looking at s.f.ratio and graduation rate
    4. Private and s.f.ratio
    5. Out of state and room and board
    6. Compare schools
    """
    #Get the data and get the Private school data to compare with the applications of students.
    data = pd.read_csv('college.csv')
    college = data.groupby('Private')
    college[['Apps','Accept','Enroll']].mean().plot(kind='barh',title=r'Student Applications',tick_label = ['Private Schools','Public Schools'])
    plt.xlabel('Amount of Applications')
    plt.legend(['Applications','Accepted Students','Enrolled Students'])
    
    #Plot a histogram of the private room-board data of price by the amount of people
    college.get_group('Yes').plot(kind='hist', y='Room.Board', legend='False',title='Private Room Board Data')
    plt.xlabel('Price ($)')
    plt.ylabel('Amount of Students')
    plt.legend(['Room & Board'])
    
    #Plot the correlation between the student faculty ratio and the graduation rate
    data.plot(kind='Hexbin',x = 'S.F.Ratio', y='Grad.Rate', gridsize=20, title= 'Correlation between Graduation Rate and S.F. Ratio')
    plt.xlabel('S.F. Ratio')
    
    plt.show()
    return
    raise NotImplementedError("Problem 4 Incomplete")







