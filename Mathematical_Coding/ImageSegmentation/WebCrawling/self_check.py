import numpy as np
import pickle

def self_check():

    # Problem 1
    with open('ans1', 'rb') as fp:
        res = pickle.load(fp)
    assert type(res) == float

    # Problem2
    """
        extract the assets for 
        JPMORGAN CHASE BK NA
        BANK OF AMER NA
        WELLS FARGO BK NA
    """
    with open('ans2', 'rb') as fp:
        res = pickle.load(fp)
    assert type(res) == list
    assert len(res) == 3
    assert type(res[0]) == list
    assert len(res[0]) == 15
    assert type(res[0][0]) == int

    # Problem 3
    """ 
        make sure your first tuple in the list 
        aka res[0] corresponds to the season that ended in 2019, 
        I won't be checking how you stored the year I'll just be 
        expecting them to be in order with the last entry being 
        the season that ended in 2010 
    """
    with open('ans3', 'rb') as fp:
        res = pickle.load(fp)
    assert type(res) == list
    assert len(res) == 10
    assert type(res[0]) == tuple
    assert len(res[0]) == 3
    assert type(res[0][1]) == str    #player's name
    assert type(res[0][2]) == int    #points scored


    #Problem 4
    """ 
        the first movie in the list should be the movie with 
        the most money or that grossed the most money, 
        not sure how to say it 
    """
    with open('ans4', 'rb') as fp:
        res = pickle.load(fp)
    assert type(res) == list
    assert len(res) == 10
    assert (type(res[0]) == str or type(res[0]) == np.str_)


    #Problem 5
    """ 
        for this problem store the list that you get when 
        the default 'linkedin' search query is passed in. the first 
        element in your list should be 'https://arxiv.org/pdf/1907.12549' 
    """
    with open('ans5', 'rb') as fp:
        res = pickle.load(fp)
    assert type(res) == list
    assert len(res) > 100
    assert len(res) < 150
    assert type(res[0]) == str

    return True
