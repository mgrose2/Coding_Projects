# regular_expressions.py
"""Volume 3: Regular Expressions.
<Mark Rose>
<Section 2>
<9/3/19>
"""
import re

# Problem 1
def prob1():
    """Compile and return a regular expression pattern object with the
    pattern string "python".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    #Create and compile a python regular expression
    pattern = re.compile("python")
    return(pattern)
    raise NotImplementedError("Problem 1 Incomplete")

# Problem 2
def prob2():
    """Compile and return a regular expression pattern object that matches
    the string "^{@}(?)[%]{.}(*)[_]{&}$".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    #Compile a regular expression for the string below
    my_match = r"\^\{\@\}\(\?\)\[\%\]\{\.\}\(\*\)\[\_\]\{\&\}\$"
    pattern = re.compile(my_match)
    return(pattern)
    raise NotImplementedError("Problem 2 Incomplete")

# Problem 3
def prob3():
    """Compile and return a regular expression pattern object that matches
    the following strings (and no other strings).

        Book store          Mattress store          Grocery store
        Book supplier       Mattress supplier       Grocery supplier

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    
    """
    #Compile a reg exp that only takes book or grocery or mattress and store or supplier
    sup = re.compile(r"^(Book|Grocery|Mattress) (store|supplier)$")
    return(sup)
    raise NotImplementedError("Problem 3 Incomplete")

# Problem 4
def prob4():
    """Compile and return a regular expression pattern object that matches
    any valid Python identifier.

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    #Make sure to add in the ? at the end so that the end part is optional
    return re.compile(r"^[a-zA-Z_]\w*\s*(\=\s*([0-9]*(.[0-9]*)?|[a-zA-Z_]\w*|\'[^\']*\'))?$")
    raise NotImplementedError("Problem 4 Incomplete")

# Problem 5
def prob5(code):
    """Use regular expressions to place colons in the appropriate spots of the
    input string, representing Python code. You may assume that every possible
    colon is missing in the input string.

    Parameters:
        code (str): a string of Python code without any colons.

    Returns:
        (str): code, but with the colons inserted in the right places.
    """
    #Create a regular expression using all the different python code inputs
    pattern1 = re.compile(r"((if|elif|else|for|while|try|except|finally|with|def|class) ?.*)")
    return pattern1.sub(r"\1:", code)
    raise NotImplementedError("Problem 5 Incomplete")

# Problem 6
def prob6(filename="fake_contacts.txt"):
    """Use regular expressions to parse the data in the given file and format
    it uniformly, writing birthdays as mm/dd/yyyy and phone numbers as
    (xxx)xxx-xxxx. Construct a dictionary where the key is the name of an
    individual and the value is another dictionary containing their
    information. Each of these inner dictionaries should have the keys
    "birthday", "email", and "phone". In the case of missing data, map the key
    to None.

    Returns:
        (dict): a dictionary mapping names to a dictionary of personal info.
    """
    #Read in the data
    with open(filename, 'r') as my_file:
        contact_data = my_file.readlines()
    
    #Create regular expressions to find the correct information
    name_pat = re.compile(r"([A-Z][a-zA-Z]* (?:[A-Z]\. )?[A-Z][a-zA-Z]*)")
    birth_pat = re.compile(r"([\d]{1,2}/[\d]{1,2}/[\d]{2,4})")
    email_pat = re.compile(r"(\S*@\S*)")
    phone_pat = re.compile(r"[0-9\(\)-]{12,}")
    
    #Create changes for all of the birthdays
    b1change = re.compile(r"^([\d]/[\d]{1,2}/[\d]{2,4})$")
    b2change = re.compile(r"^([\d]{2})/([\d]/[\d]{2,4})$")
    b3change = re.compile(r"^([\d]{2}/[\d]{2})/([\d]{2})$")
    
    #Create changes for all the phone numbers
    p1change = re.compile(r"([\d]{3})(?:[-\)]{1,2})")
    p2change = re.compile(r"[\d]{4}")
    my_dic = {}
    
    #Iterate through the data
    for i in contact_data:
        name = name_pat.findall(i)[0]
        
        #Get and change the birthday if it exists
        if not birth_pat.findall(i):
            birthday = None
        else:
            birthday = birth_pat.findall(i)[0]
            birthday = b1change.sub(r"0\1", birthday)
            birthday = b2change.sub(r"\1/0\2", birthday)
            birthday = b3change.sub(r"\1/20\2", birthday)
        
        #Get and save the email as an attribute
        if not email_pat.findall(i):
            email = None
        else:
            email = email_pat.findall(i)[0]
           
        #Get and change the phone numbers into the proper order
        if not phone_pat.findall(i):
            phone = None
        else:
            phone = phone_pat.findall(i)[0]
            first_two = p1change.findall(phone)
            last = p2change.findall(phone)
            phone = "(" + first_two[0] + ")" + first_two[1] + "-" + last[0]
           
        #Define the dictionary 
        my_dic[name]= {"birthday": birthday, "email": email, "phone": phone}
        
    #Return the dictionary
    return(my_dic)

    raise NotImplementedError("Problem 6 Incomplete")
