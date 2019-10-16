# sql1.py
"""Volume 3: SQL 1 (Introduction).
<Mark Rose>
<Section 2>
<11/14/18>
"""
import sqlite3 as sql
import csv
import numpy as np
import matplotlib.pyplot as plt

# Problems 1, 2, and 4
def student_db(db_file="students.db", student_info="student_info.csv",
                                      student_grades="student_grades.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the tables MajorInfo, CourseInfo, StudentInfo, and StudentGrades from
    the database (if they exist). Recreate the following (empty) tables in the
    database with the specified columns.

        - MajorInfo: MajorID (integers) and MajorName (strings).
        - CourseInfo: CourseID (integers) and CourseName (strings).
        - StudentInfo: StudentID (integers), StudentName (strings), and
            MajorID (integers).
        - StudentGrades: StudentID (integers), CourseID (integers), and
            Grade (strings).

    Next, populate the new tables with the following data and the data in
    the specified 'student_info' 'student_grades' files.

                MajorInfo                         CourseInfo
            MajorID | MajorName               CourseID | CourseName
            -------------------               ---------------------
                1   | Math                        1    | Calculus
                2   | Science                     2    | English
                3   | Writing                     3    | Pottery
                4   | Art                         4    | History

    Finally, in the StudentInfo table, replace values of −1 in the MajorID
    column with NULL values.

    Parameters:
        db_file (str): The name of the database file.
        student_info (str): The name of a csv file containing data for the
            StudentInfo table.
        student_grades (str): The name of a csv file containing data for the
            StudentGrades table.
    """
    with open(student_info, 'r') as infile:
        studentinfo_rows = list(csv.reader(infile))
    with open(student_grades, 'r') as infile:
        studentgrades_rows = list(csv.reader(infile))                                        #save and load the data
    
    majorinfo_rows = [(1, "Math"), (2, "Science"), (3, "Writing"), (4, "Art")]				#create data for tables
    courseinfo_rows = [(1, "Calculus"), (2, "English"), (3, "Pottery"), (4, "History")]
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            cur.execute("DROP TABLE IF EXISTS MajorInfo")                                   #drop the tables if the exist
            cur.execute("DROP TABLE IF EXISTS CourseInfo")
            cur.execute("DROP TABLE IF EXISTS StudentInfo")
            cur.execute("DROP TABLE IF EXISTS StudentGrades")
            cur.execute("CREATE TABLE MajorInfo (MajorID INTEGER, MajorName TEXT)")
            cur.execute("CREATE TABLE CourseInfo (CourseID INTEGER, CourseName TEXT)")
            cur.execute("CREATE TABLE StudentInfo (StudentID INTEGER, StudentName TEXT, MajorID INTEGER)")
            cur.execute("CREATE TABLE StudentGrades (StudentID INTEGER, CourseID INTEGER ,Grade TEXT)")
            cur.executemany("INSERT INTO MajorInfo VALUES(?,?);", majorinfo_rows)		#insert data into tables
            cur.executemany("INSERT INTO CourseInfo VALUES(?,?);", courseinfo_rows)
            cur.executemany("INSERT INTO StudentInfo VALUES(?,?,?);", studentinfo_rows)
            cur.executemany("INSERT INTO StudentGrades VALUES(?,?,?);", studentgrades_rows)
            cur.execute("UPDATE StudentInfo SET MajorID=NULL WHERE MajorID==-1;")
    finally:
        conn.close()
    return
    
    raise NotImplementedError("Problem 1 Incomplete")


# Problems 3 and 4
def earthquakes_db(db_file="earthquakes.db", data_file="us_earthquakes.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the USEarthquakes table if it already exists, then create a new
    USEarthquakes table with schema
    (Year, Month, Day, Hour, Minute, Second, Latitude, Longitude, Magnitude).
    Populate the table with the data from 'data_file'.

    For the Minute, Hour, Second, and Day columns in the USEarthquakes table,
    change all zero values to NULL. These are values where the data originally
    was not provided.

    Parameters:
        db_file (str): The name of the database file.
        data_file (str): The name of a csv file containing data for the
            USEarthquakes table.
    """
    with open(data_file, 'r') as infile:			#save data
        rows = list(csv.reader(infile))
    
    with sql.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS USEarthquakes")		                                                                     #drop the table if exists
        cur.execute("CREATE TABLE USEarthquakes (Year INTEGER, Month INTEGER, Day INTEGER, " 		                                 #create table
                    "Hour INTEGER, Minute INTEGER, Second INTEGER, Latitude REAL, Longitude REAL, Magnitude REAL)")
        cur.executemany("INSERT INTO USEarthquakes VALUES(?,?,?,?,?,?,?,?,?);", rows)		                                         #insert data into table
        cur.execute("DELETE FROM USEarthquakes WHERE Magnitude==0;")		                                                         #delete 0 values from table
        cur.execute("UPDATE USEarthquakes SET Day=NULL WHERE Day==0;")		                                                         #and replace them with NULL
        cur.execute("UPDATE USEarthquakes SET Hour=NULL WHERE Hour==0;")
        cur.execute("UPDATE USEarthquakes SET Minute=NULL WHERE Minute==0;")
        cur.execute("UPDATE USEarthquakes SET Second=NULL WHERE Second==0;")
        
    conn.close()
    return 

    raise NotImplementedError("Problem 3 Incomplete")


# Problem 5
def prob5(db_file="students.db"):
    """Query the database for all tuples of the form (StudentName, CourseName)
    where that student has an 'A' or 'A+'' grade in that course. Return the
    list of tuples.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    with sql.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute("SELECT SI.StudentName, CI.CourseName "			                                                                   #Select Name and Course Name
                    "FROM StudentInfo AS SI, CourseInfo AS CI, StudentGrades as SG "	                                               #Where student has an A or A+
                    "WHERE SI.StudentID == SG.StudentID AND SG.CourseID==CI.CourseID AND SG.Grade IN ('A', 'A+');")
        results = cur.fetchall()			                                                                                           #get all the results
    conn.close()
    return results
    
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def prob6(db_file="earthquakes.db"):
    """Create a single figure with two subplots: a histogram of the magnitudes
    of the earthquakes from 1800-1900, and a histogram of the magnitudes of the
    earthquakes from 1900-2000. Also calculate and return the average magnitude
    of all of the earthquakes in the database.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (float): The average magnitude of all earthquakes in the database.
    """
    
    earthquakes_db()
    
    with sql.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute("SELECT Magnitude "					                                            #select magnitudes from 1800-1900
                    "FROM USEarthquakes "
                    "WHERE Year BETWEEN 1800 AND 1899;")
        nineteenth = np.ravel(cur.fetchall())			                                            #get all the results
            
        
        cur.execute("SELECT Magnitude "					                                            #select magnitudes from 1900-2000
                    "FROM USEarthquakes "
                    "WHERE Year BETWEEN 1900 AND 1999;")
        twentieth = np.ravel(cur.fetchall())			                                            #get all the results
        
        cur.execute("SELECT AVG(Magnitude) FROM USEarthquakes;")
        average = cur.fetchall()						                                            #get the average magnitude
        
    conn.close()
    
    ax1 = plt.subplot(121)								                                            #build the graph
    ax1.hist(nineteenth)								                                            #graph the 19th century magnitudes
    ax1.set_title("19th Century")
    plt.xlabel("Magnitude")
    plt.ylabel("Number of Earthquakes")
    ax2 = plt.subplot(122)
    ax2.hist(twentieth)									                                            #graph the 20th century magnitudes
    ax2.set_title("20th Century")   
    plt.xlabel("Magnitude")
    plt.show()

    
    return np.ravel(average)							#return the average magnitude
    raise NotImplementedError("Problem 6 Incomplete")
