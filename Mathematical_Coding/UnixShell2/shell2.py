# shell2.py
"""Volume 3: Unix Shell 2.
<Mark Rose>
<Section 2>
<4/1/19>
"""

import os
from glob import glob
import numpy as np

# Problem 5
def grep(target_string, file_pattern):
    """Find all files in the current directory or its subdirectories that
    match the file pattern, then determine which ones contain the target
    string.

    Parameters:
        target_string (str): A string to search for in the files whose names
            match the file_pattern.
        file_pattern (str): Specifies which files to search.
    """
    targets = []											#initialize targets
    files = glob("**/" + file_pattern, recursive=True)		#glob the files
    for file in files:
        if target_string in open(file).read():				#search for string in files
            targets.append(file)
            
    return targets											#return targets
        


# Problem 6
def largest_files(n):
    """Return a list of the n largest files in the current directory or its
    subdirectories (from largest to smallest).
    """
    files = glob('**/*.*', recursive=True)					#glob the files
    size = []
    for i in files:											#search through files
        size.append(os.path.getsize(i))						#append size of files
        
    order = np.argsort(size)								#sort file sizes
    files = np.array(files)
    return list(files[order][::-1][:n])						#return n largest files
