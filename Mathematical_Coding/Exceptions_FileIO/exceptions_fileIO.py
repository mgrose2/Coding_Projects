# exceptions_fileIO.py
"""Python Essentials: Exceptions and File Input/Output.
<Mark Rose>
<Section 2>
<9/26/18>
"""

from random import choice


# Problem 1
def arithmagic():
    """A fuction that is implemented to do some arithmetic magic. """
    step_1 = input("Enter a 3-digit number where the first and last "
                                           "digits differ by 2 or more: ")
    if len(step_1) != 3:
        raise ValueError('The number must be a 3-digit integer.')    #This is to make sure that there is correct input.
    if abs(int(step_1[0])-int(step_1[2])) < 2:
        raise ValueError('The difference between the first and the last digit must be two or more.')  #The rest of the ValueErrors are to correct human error.
    step_2 = input("Enter the reverse of the first number, obtained by reading it backwards: ")
    if step_2 != step_1[::-1]:
        raise ValueError('The number must be the reverse of the original number.')
    step_3 = input("Enter the positive difference of these numbers: ")
    if int(step_3) != abs(int(step_1)-int(step_2)):
        raise ValueError('The number must be the positive difference of the original number and its reverse.')
    step_4 = input("Enter the reverse of the previous result: ")
    if step_4 != step_3[::-1]:
        raise ValueError('The number must be the reverse of the previous number.')
    print(str(step_3), "+", str(step_4), "= 1089 (ta-da!)")

# Problem 2
def random_walk(max_iters=1e12):
    """This function is designed to act as a game where one can walk in two directions. One can control c to stop the walk and see what point you are at."""
    walk = 0
    directions = [1, -1]
    try:
        for i in range(int(max_iters)):
            walk += choice(directions)
    except KeyboardInterrupt:
        print('Process terminated at iteration', i)  #Prints where the walk ended and returns the value if there is a control c keyboard interruption.
        return walk
    print("Process Completed")          #If completed, says so and returns the value
    return walk

# Problems 3 and 4: Write a 'ContentFilter' class.
class ContentFilter():
    """Set the file name for Content Filter
    Parameters:
        file_name (str): The name of the file to be read.
        access (str): The type of access for the file
    """   
    def __init__(self, file_name):
        """Initializing the ContentFilter class with a file name. Only read in when there is a valid name."""
        while True:
            try:
                my_file = open(file_name, 'r')          #Try to open the file
                self.file_name = file_name              #If the file opens, initialize the ContentFilter object
                self.contents = my_file.read()
                my_file.close()
                break
            except Exception as e:
                file_name = input("Please enter a valid file name: ")   #If the file doesn't open, ask for good input.
        
        
    def uniform(self, outfile, mode='w', case = 'upper'):
        """A ContentFilter function that is used to either set all of the letters in a file to uppercase or lowercase, and then write it to a new or existing file"""
        if mode != 'w' and mode != 'x' and mode != 'a':
            raise ValueError("Your access method must be 'x', 'w', or 'a'")
        if case == "upper":                                             #Write the file's contents in all upper case or lower case depending on the input
            with open(outfile, mode) as myfile:
                myfile.write(self.contents.upper())
        elif case == "lower":
            with open(outfile, mode) as myfile:
                myfile.write(self.contents.lower())
        else:
            raise ValueError("Your input must be 'lower' or 'upper'")
        
    def reverse(self, outfile, mode='w', unit = 'line'):
        """A ContentFilter function that is used to reverse the words in a line, or reverse the lines and then write it to a new or existing file"""
        if mode != 'w' and mode != 'x' and mode != 'a':
            raise ValueError("Your access method must be 'x', 'w', or 'a'")
        if unit == "line":                                          #Reverse the order of lines if unit = 'line' or the words if unit = 'word'
            with open(outfile, mode) as myfile:
                r = self.contents.splitlines()                      #Split the text into a list of lines
                for i in r[::-1]:
                    if i != '\n':
                        myfile.write(i+'\n')
        elif unit == "word":
            with open(outfile, mode) as myfile:
                r = self.contents.splitlines()
                for line in r:
                    j = line.split(' ')                             #Split each line into a list of words
                    for word in j[::-1]:
                        myfile.write(word)
                        myfile.write(' ')
                    if line != '\n':
                        myfile.write('\n')                          
        else:
            raise ValueError("Your input must be 'unit' or 'line'")
            
    def transpose(self, outfile, mode='w'):
        """A ContentFilter function that is used to write the first word of each line to be the first line, and the second word of each line to be the second line, etc."""
        if mode != 'w' and mode != 'x' and mode != 'a':
            raise ValueError("Your access method must be 'x', 'w', or 'a'")
        else:
            with open(outfile, mode) as myfile:
                r = self.contents.splitlines()                  #Split the text into a list of lines
                for i in range(len(r)):
                    r[i] = r[i].split(' ')                      #Split each line into a list of words
                for i in range(len(r[0])):
                    for j in range(len(r)):                     #Loop through each line and write the first word to the first line, second word to the second line, and so on
                        myfile.write(r[j][i]+' ')
                    myfile.write('\n')
            
                        
    def __str__(self):
        """An overridden string function for the ContentFilter class. This prints out the file name, total characters, alphabetic characters, numerical characters, whitespace characters, and the number of lines."""
        r = self.contents.splitlines()
        total_lines = len(r)
        white_space = sum(c.isspace() for c in self.contents )          #Checks which values are '\t' or '\n' or ' ' and sums up the amount that are true
        numbers = sum(c.isdigit() for c in self.contents)
        letters   = sum(c.isalpha() for c in self.contents)             #Letters and numbers are similar to how the white space is counted up
        total_characters = white_space + numbers + letters
        file_name = self.file_name
        
        return('Source file:\t\t\t' + file_name + '\n'
             + 'Total characters:\t\t' + str(total_characters) + '\n'       #returns a string of each of the found variables
             + 'Alphabetic characters:\t\t' + str(letters) + '\n'
             + 'Numerical characters:\t\t' + str(numbers) + '\n'
             + 'Whitespace characters:\t\t' + str(white_space) + '\n'
             + 'Number of lines:\t\t' + str(total_lines)
             )
