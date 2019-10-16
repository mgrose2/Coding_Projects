# solutions.py
"""Volume 3: SQL 2.
<Mark Rose>
<Section 2>
<11/28/18>
"""

import sqlite3 as sql

# Problem 1
def prob1(db_file="students.db"):
    """Query the database for the list of the names of students who have a
    'B' grade in any course. Return the list.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): a list of strings, each of which is a student name.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()
    names = cur.execute("SELECT SI.StudentName "                                                #Get the student name
                        "FROM StudentInfo AS SI INNER JOIN StudentGrades AS MI "                #Join the two tables StudentInfo and Student Grades
                        "ON SI.StudentID = MI.StudentID "                                       #Join them where their StudentID's are the same
                        "WHERE MI.Grade == 'B';").fetchall()                                    #Get the grades that are a B
    return([names[0] for names in names])
    
    raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def prob2(db_file="students.db"):
    """Query the database for all tuples of the form (Name, MajorName, Grade)
    where 'Name' is a student's name and 'Grade' is their grade in Calculus.
    Only include results for students that are actually taking Calculus, but
    be careful not to exclude students who haven't declared a major.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()
    info = cur.execute("SELECT SI.StudentName, MI.MajorName, SG.Grade "                     #Get the student name, major name, and grade they are getting in Calculus
                        "FROM StudentInfo AS SI LEFT OUTER JOIN MajorInfo AS MI "           #Outer join the student info and major info so as to not exclude undeclared majors
                        "ON SI.MajorID == MI.MajorID "
                        "INNER JOIN StudentGrades AS SG "                                   #Inner join the table with the student grades
                        "ON SI.StudentID == SG.StudentID "
                        "WHERE SG.CourseID == '1';").fetchall()                             #Get only the grades where the course ID equates to calculus
    return info
    raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def prob3(db_file="students.db"):
    """Query the database for the list of the names of courses that have at
    least 5 students enrolled in them.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        ((list): a list of strings, each of which is a course name.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()
    info = cur.execute("SELECT CI.CourseName "                                             #Get the course name
                       "FROM StudentGrades AS SG INNER JOIN CourseInfo as CI "             #Inner join the student graes and courseinfo tables
                       "ON SG.CourseID == CI.CourseID "                                    #Join them where their course ids are the same
                       "GROUP BY CI.CourseName "                                           #Group them all by their course name
                       "HAVING COUNT(*) >= 5").fetchall()                                  #Return the counts that are at least 5
    return([info[0] for info in info])
    raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def prob4(db_file="students.db"):
    """Query the given database for tuples of the form (MajorName, N) where N
    is the number of students in the specified major. Sort the results in
    descending order by the counts N, then in alphabetic order by MajorName.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()
    info =cur.execute("SELECT MI.MajorName, COUNT(*) as num_students "                          #Get the major name and the number of students in each major
                       "FROM StudentInfo as SI LEFT OUTER JOIN MajorInfo as MI "                     #Join the tables student info and major info
                       "ON SI.MajorID == MI.MajorID "                                           #Join where the major ids are the same
                       "GROUP BY MI.MajorName "
                       "ORDER BY num_students DESC, MI.MajorName ASC").fetchall()               #Return the number in each course in descending order, then in alphebetic order by major name
    return(info)
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def prob5(db_file="students.db"):
    """Query the database for tuples of the form (StudentName, MajorName) where
    the last name of the specified student begins with the letter C.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()
    info = cur.execute("SELECT SI.StudentName, MI.MajorName "                                   #Get the student and their major name
                       "FROM StudentInfo as SI LEFT OUTER JOIN MajorInfo as MI "                #Outer join the student info and major info tables to not leave out Null major values
                       "ON SI.MajorID == MI.MajorID "
                       "WHERE StudentName LIKE '% C%';").fetchall()                             #Return any student name where the last name starts with a C
    return(info)
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def prob6(db_file="students.db"):
    """Query the database for tuples of the form (StudentName, N, GPA) where N
    is the number of courses that the specified student is in and 'GPA' is the
    grade point average of the specified student according to the following
    point system.

        A+, A  = 4.0    B  = 3.0    C  = 2.0    D  = 1.0
            A- = 3.7    B- = 2.7    C- = 1.7    D- = 0.7
            B+ = 3.4    C+ = 2.4    D+ = 1.4

    Order the results from greatest GPA to least.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()
    info = cur.execute("SELECT StudentName, COUNT(*) AS num_courses, AVG(Grade) as GPA "            #Get the student name, number of courses, and GPA (by taking the average of the GPAs
                       "FROM ( "
                           "SELECT StudentID, CASE Grade "                                          #Selecte the StudentIDs and make a case with the grade
                                "WHEN 'A+' THEN '4.0' "                                             #Assign the grades various values based on the given score system
                                "WHEN 'A' THEN '4.0' "
                                "WHEN 'A-' THEN '3.7' "
                                "WHEN 'B+' THEN '3.4' "
                                "WHEN 'B' THEN '3.0' "
                                "WHEN 'B-' THEN '2.7' "
                                "WHEN 'C+' THEN '2.4' "
                                "WHEN 'C' THEN '2.0' "
                                "WHEN 'C-' THEN '1.7' "
                                "WHEN 'D+' THEN '1.4' "
                                "WHEN 'D' THEN '1.0' "
                                "WHEN 'D-' THEN '0.7' END as Grade "                               #Rename the new column Grade
                            "FROM StudentGrades) as CG INNER JOIN StudentInfo as SI "              #Inner join with student info so we can get the names
                        "ON CG.StudentID == SI.StudentID "  
                        "GROUP BY CG.StudentID "                                                   #Use the grouping function to get the number of courses
                        "ORDER BY GPA DESC;").fetchall()                                           #Order from highest GPA to least
    return(info)
                           
    raise NotImplementedError("Problem 6 Incomplete")
