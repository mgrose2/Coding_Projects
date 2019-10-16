# python_intro.py
"""Python Essentials: Introduction to Python.
<Mark Rose>
<Math 345>
<9/4/18>
"""

def sphere_volume(r):
    """This is a function to get the volume of a sphere. Take a radius and compute"""
    return(3.14159*r**3*4/3) 

def isolate(a,b,c,d,e):
    """Practicing spaces"""
    print(a, b,c, sep='     ', end = ' ')
    print(d, e, sep=' ') 

def pig_latin(word):
    """take a word and translate it into pig latin"""
    for i in {'a','e','i','o','u'}:
        if i == word[0]:
            word = word + 'hay'
            return(word)
    else:
        word = word[1:len(word)]+word[0]+'ay' 
        return(word) 

def palindrome():
    maxer = 0
    for i in range(100,1000):
        for j in range(100,1000):
            number = i*j          
            if str(number) == str(number)[::-1] and number > maxer:
                maxer = number
    return(maxer)

def list_ops():
    my_list = ['bear', 'ant', 'cat', 'dog']
    my_list.append('eagle')
    my_list[2] = 'fox'
    my_list.pop(1)
    my_list.sort(reverse=True)
    my_list[my_list.index('eagle')] = 'hawk'
    my_list[-1] = my_list[-1] + 'hunter'
    return(my_list)
       
def first_half(a):    
    return(a[0:int(len(a)/2)])

def backward(a):
    return(a[::-1])

def alt_harmonic(n):
    my_list = [(-1)**(i+1)/i for i in range(1,n+1)]
    summer = sum(my_list)
    return(summer)                      

if __name__ == "__main__":
    print('Hello, world!')
