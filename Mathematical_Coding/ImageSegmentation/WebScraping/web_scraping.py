"""Volume 3: Web Scraping.
<Mark Rose>
<Section 2>
<9/16/19>
"""
import requests
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# Problem 1
def prob1():
    """Use the \li{requests} library to get the HTML source for the website http://www.example.com.
    Save the source as a file called example.html.
    If the file already exists, do not scrape the website or overwrite the file.
    """
    #Get the file from example.com
    outfile = "example.html"
    
    #Create the file if it does't exist
    if not os.path.exists(outfile):
        response = requests.get('http://example.com')
        with open (outfile,'w') as myfile:
            myfile.write(response.text)
    else:
        print("File already exists")
    return
    raise NotImplementedError("Problem 1 Incomplete")
    
# Problem 2
def prob2():
    """Examine the source code of http://www.example.com. Determine the names
    of the tags in the code and the value of the 'type' attribute associated
    with the 'style' tag.

    Returns:
        (set): A set of strings, each of which is the name of a tag.
        (str): The value of the 'type' attribute in the 'style' tag.
    """
    #Read example.html which was made in prob1
    infile = "example.html"
    with open(infile, 'r') as myfile:
        text = myfile.read()
    new_tags = []
    
    #Create a regex pattern to match a tag and a regex pattern to find the value of the type attribute 
    tag_pattern = re.compile(r'(?:<)([a-zA-Z0-9]+)(?:[^>]*>)')
    style_pattern = re.compile(r'<style type=\"([a-zA-Z]*/[a-zA-Z]*)\">')
    
    #Match within the text the style type and tags
    style_str = re.findall(style_pattern, text)
    tags = re.findall(tag_pattern, text)
    
    #Make sure the tags are in the right form and return
    return(set(tags), style_str[0])
    
    raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def prob3(code):
    """Return a list of the names of the tags in the given HTML code."""
    #Read in code
    #infile = code
    #with open(infile, 'r') as myfile:
    #    text = myfile.read()

    #Get all the tags from the text
    soup = BeautifulSoup(code, 'html.parser')
    tags = soup.findAll(True)

    #Create a list of names of the tags
    names = []
    for i in range(len(tags)):
        names.append(tags[i].name)
    return names


# Problem 4
def prob4(filename="example.html"):
    """Read the specified file and load it into BeautifulSoup. Find the only
    <a> tag with a hyperlink and return its text.
    """
    #Read in code
    infile = filename
    with open(infile, 'r') as myfile:
        text = myfile.read()

    #Get the text from the a tag
    soup = BeautifulSoup(text, 'html.parser')
    tags = soup.find_all(name='a')
    a_tag = soup.a

    return a_tag.text


# Problem 5
def prob5(filename="san_diego_weather.html"):
    """Read the specified file and load it into BeautifulSoup. Return a list
    of the following tags:

    1. The tag containing the date 'Thursday, January 1, 2015'.
    2. The tags which contain the links 'Previous Day' and 'Next Day'.
    3. The tag which contains the number associated with the Actual Max
        Temperature.

    Returns:
        (list) A list of bs4.element.Tag objects (NOT text).
    """

    #Read in code
    infile = filename
    with open(infile, 'r') as myfile:
        text = myfile.read()

    #Load the text and find the tags
    soup = BeautifulSoup(text, 'html.parser')
    tags = []
    tags.append(soup.find(string='Thursday, January 1, 2015').parent)
    tags.append(soup.find(href='/history/airport/KSAN/2015/1/2/DailyHistory.html'))
    tags.append(soup.find(href='/history/airport/KSAN/2014/12/31/DailyHistory.html'))
    tags.append(soup.find_all(class_='wx-value')[2])

    return tags

# Problem 6
def prob6(filename="large_banks_index.html"):
    """Read the specified file and load it into BeautifulSoup. Return a list
    of the tags containing the links to bank data from September 30, 2003 to
    December 31, 2014, where the dates are in reverse chronological order.

    Returns:
        (list): A list of bs4.element.Tag objects (NOT text).
    """
    #Read in code
    infile = filename
    with open(infile, 'r') as myfile:
        text = myfile.read()

    #Load the text and find the tags
    date_pattern = re.compile(r'20(0[3-9])|(1[0-4])')
    soup = BeautifulSoup(text, 'html.parser')

    return [tag.parent for tag in soup.find_all(string=date_pattern)][:-2]
    raise NotImplementedError("Problem 6 Incomplete")


# Problem 7
def prob7(filename="large_banks_data.html"):
    """Read the specified file and load it into BeautifulSoup. Create a single
    figure with two subplots:

    1. A sorted bar chart of the seven banks with the most domestic branches.
    2. A sorted bar chart of the seven banks with the most foreign branches.

    In the case of a tie, sort the banks alphabetically by name.
    """
    #Read in code and clean the data
    html = pd.read_html(filename)[2]
    vals1 = html['Domestic Branches'].fillna('0')
    vals1 = [int(val.replace(',', '')) for val in vals1.replace('.', '0')]
    html['Domestic Branches'] = vals1

    vals2 = html['Foreign Branches'].fillna('0')
    vals2 = [int(val.replace(',', '')) for val in vals2.replace('.', '0')]
    html['Foreign Branches'] = vals2

    #Get the seven banks with the most domestic and foreign branches
    domestic = html.sort_values(by=['Domestic Branches'], ascending=False).head(7)
    foreign = html.sort_values(by=['Foreign Branches'], ascending=False).head(7)

    #Put the data into lists to graph
    domestic_banks = domestic['Bank Name / Holding Co Name'].tolist()
    domestic_branches = domestic['Domestic Branches'].tolist()
    foreign_banks = foreign['Bank Name / Holding Co Name'].tolist()
    foreign_branches = foreign['Foreign Branches'].tolist()

    #Graph the data
    fig, ax = plt.subplots(2)
    ax[0].barh(domestic_banks, domestic_branches)
    ax[1].barh(foreign_banks, foreign_branches)
    ax[0].set_title('Domestic Branches')
    ax[1].set_title('Foreign Branches')
    ax[0].set_xlabel('Number of Branches')
    ax[1].set_xlabel('Number of Branches')
    ax[0].set_ylabel('Name of Bank')
    ax[1].set_ylabel('Name of Bank')
    plt.tight_layout()

    plt.show()

    return
    raise NotImplementedError("Problem 7 Incomplete")

