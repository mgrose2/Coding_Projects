# standard_library.py
"""Python Essentials: The Standard Library.
<Mark Rose>
<320>
<9/12/18>
"""

import calculator as calc
from itertools import combinations
import sys
import random
import box
import time

# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order).
    """
    return(min(L), max(L), sum(L)/len(L))
    raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test numbers, strings, lists, tuples, and sets. Print your results.
    """
    number = 4
    check_num = number
    check_num = 5
    if number == check_num:
        print('Ints are mutable')
    else:
        print('Ints are immutable')

    word = 'mark'
    check_word = word
    check_word = check_word.capitalize()
    if word == check_word:
        print('Strings are mutable')
    else:
        print('Strings are immutable')

    my_list = list(word)
    check_list = my_list
    check_list[0] = 'Q'
    if my_list == check_list:
        print('Lists are mutable')
    else:
        print('Lists are immutable')

    my_tup = (1,2)
    check_tup = my_tup
    check_tup += (1,)
    if my_tup == check_tup:
        print('Tuples are mutable')
    else:
        print('Tuples are immutable')

    my_set = set(word)
    check_set = my_set
    check_set.add('u')
    if my_set == check_set:
        print('Sets are mutable')
    else:
        print('Sets are immutable')
    return
    raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than those that are imported from your
    'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """
    return(calc.squareroot(calc.multiply(a, a) + calc.multiply(b,b)))
    raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """
    my_list = []
    for i in range(0,len(A)+1):    
        new_list = combinations(A, i)
        my_list.extend(new_list)
    for i in range(len(my_list)):
        my_list[i] = set(my_list[i])
    return(my_list)
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5: Implement shut the box.
def shut_the_box(name, time_left):
    remain_num = list(range(1,10))
    dice_list = list(range(1,6))
    roll = 0
    time_left = int(time_left)
    possible = True
    while(possible):
        start_time = time.time()
        good_job = True
        if min(remain_num)+max(remain_num) < 7:
            roll = random.choice(dice_list)
        else:
            roll = random.choice(dice_list)+random.choice(dice_list)  
        print('Numbers Left: ', remain_num)
        print('Roll: ', roll)
        if box.isvalid(roll, remain_num) == False:
            print('Game Over')
            return
        print('Seconds Left: ', round(time_left))
        while good_job == True:
            elim = input('Numbers To Eliminate: ')
            move_on = box.parse_input(elim, remain_num)
            if not move_on:
                print('Invalid input. Please try again.')
            elif not sum(move_on) == roll:
                print('Fix your addition. Please try again.')
            else:
                for i in move_on:
                    remain_num.remove(i) 
                good_job = False      
        end_time = time.time()
        time_left = time_left - (end_time-start_time)
        if (time_left <= 0):
            print('Game Over')
            return
        if not remain_num:
            possible = False
    print('You win with ', time_left, ' seconds remaining!!')

if __name__ == "__main__":
    if len(sys.argv) == 3:
        shut_the_box(sys.argv[1],sys.argv[2])
    
