# test_specs.py
"""Python Essentials: Unit Testing.
<Mark Rose
<Section 2>
<10/22/18>
"""

import specs
import pytest


def test_add():
    assert specs.add(1, 3) == 4, "failed on positive integers"
    assert specs.add(-5, -7) == -12, "failed on negative integers"
    assert specs.add(-6, 14) == 8

def test_divide():
    assert specs.divide(4,2) == 2, "integer division"
    assert specs.divide(5,4) == 1.25, "float division"
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.divide(4, 0)
    assert excinfo.value.args[0] == "second input cannot be zero"


# Problem 1: write a unit test for specs.smallest_factor(), then correct it.
def test_smallest_factor():
    assert specs.smallest_factor(1) == 1, "smallest factor of 1 should be 1"
    assert specs.smallest_factor(2) == 2, "should return itself"
    assert specs.smallest_factor(3) == 3, "should return itself"
    assert specs.smallest_factor(4) == 2, "remember, 2 is a prime number"
    assert specs.smallest_factor(6) == 2, "remember, 2 is a prime number"                                   #various test cases for smallest_factor function
    assert specs.smallest_factor(77) == 7, "not working with larger primes"
    assert specs.smallest_factor(-2) == None, "Negative numbers are being accepted"
    assert specs.smallest_factor(.045) == None, "Non integers being accepted"
# Problem 2: write a unit test for specs.month_length().
def test_month_length():
    assert specs.month_length('January') == 31, "31 day months are incorrect"
    assert specs.month_length('February', False) == 28, "non leap year February is incorrect"               #tests all of the special cases for months of the year
    assert specs.month_length('February', True) == 29, "leap year February is incorrect"
    assert specs.month_length('April') == 30, "30 day months are incorrect"
    assert specs.month_length('Jay') == None, "Non months are giving incorrect answers"

# Problem 3: write a unit test for specs.operate().
def test_operate():
    with pytest.raises(TypeError) as excinfo:
        specs.operate(2, 1, 2)
    assert excinfo.value.args[0] == "oper must be a string"
    assert specs.operate(1,2,'+') == 3, "addition operator is incorrect"                                    #tests each operator
    assert specs.operate(3,2,'-') == 1, "subtraction operator is incorrect"
    assert specs.operate(3,2,'*') == 6, "multiplication operator is incorrect"
    assert specs.operate(4,2,'/') == 2, "division operator is incorrect"
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.operate(4, 0,'/')
    assert excinfo.value.args[0] == "division by zero is undefined"
    with pytest.raises(ValueError) as excinfo:
        specs.operate(4, 0, '**')
    assert excinfo.value.args[0] == "oper must be one of '+', '/', '-', or '*'"


# Problem 4: write unit tests for specs.Fraction, then correct it.
@pytest.fixture
def set_up_fractions():
    frac_1_3 = specs.Fraction(1, 3)
    frac_1_2 = specs.Fraction(1, 2)
    frac_n2_3 = specs.Fraction(-2, 3)
    return frac_1_3, frac_1_2, frac_n2_3
    
def test_fraction_init(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_3.numer == 1
    assert frac_1_2.denom == 2
    assert frac_n2_3.numer == -2
    frac = specs.Fraction(30, 42)
    assert frac.numer == 5
    assert frac.denom == 7
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.Fraction(1, 0)
    assert excinfo.value.args[0] == "denominator cannot be zero"
    with pytest.raises(TypeError) as excinfo:
        specs.Fraction(3.1, 4.5)
    assert excinfo.value.args[0] == "numerator and denominator must be integers"

def test_fraction_str(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert str(frac_1_3) == "1/3"
    assert str(frac_1_2) == "1/2"
    assert str(frac_n2_3) == "-2/3"
    assert str(specs.Fraction(3,1)) == "3"

def test_fraction_float(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert float(frac_1_3) == 1 / 3.
    assert float(frac_1_2) == .5
    assert float(frac_n2_3) == -2 / 3.

def test_fraction_eq(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 == specs.Fraction(1, 2)
    assert frac_1_3 == specs.Fraction(2, 6)
    assert frac_n2_3 == specs.Fraction(8, -12)
    assert specs.Fraction(2,1) == 2

#create fractions to use to test   
@pytest.fixture
def set_up_fractions2():
    frac_1_4 = specs.Fraction(1, 4)
    frac_1_5 = specs.Fraction(1, 5)
    return frac_1_4, frac_1_5
    
                                                                                                        #unit tests for addition
def test_fraction_add(set_up_fractions2):
    frac_1_4, frac_1_5 = set_up_fractions2
    assert frac_1_4 + frac_1_5 == specs.Fraction(9,20), "Failed on addition"

                                                                                                        #unit tests for subraction
def test_fraction_sub(set_up_fractions2):
    frac_1_4, frac_1_5 = set_up_fractions2
    assert frac_1_4 - frac_1_5 == specs.Fraction(1,20), "Failed on subtraction"

                                                                                                        #unit tests for multiplication    
def test_fraction_mul(set_up_fractions2):
    frac_1_4, frac_1_5 = set_up_fractions2
    assert frac_1_4 * frac_1_5 == specs.Fraction(1,20), "Failed on multiplication"

                                                                                                        #unit tests for division    
def test_fraction_trudiv(set_up_fractions2):
    frac_1_4, frac_1_5 = set_up_fractions2
    assert frac_1_4 / frac_1_5 == specs.Fraction(5,4), "Failed on division"
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.Fraction(1,5) / specs.Fraction(0,4)

# Problem 5: Write test cases for Set.
def test_count_sets():
                                                                                                    #test for 12 cards
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(["1111", "2222", "0000"])
    assert excinfo.value.args[0] == "list must have exactly 12 cards"
                                                                                                    #test for unique cards
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(["1022", "1122", "0100", "2021",
         "0010", "2201", "2111", "0020",
         "1102", "0210", "2110", "1022"])
    assert excinfo.value.args[0] == "the cards must all be unique"
                                                                                                    #test for 4 digits
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(["1022", "1122", "0100", "2021",
         "0010", "2201", "2111", "0020",
         "1102", "0210", "211", "102"])
    assert excinfo.value.args[0] == "all cards must have exactly 4 digits"
                                                                                                    #test for values 0,1,2
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(["1022", "1122", "0100", "2021",
         "0010", "2201", "2111", "0020",
         "1142", "0210", "2310", "1020"])
    assert excinfo.value.args[0] == "all characters must be 0, 1, or 2"
                                                                                                    #test to verify function works
    assert specs.count_sets(["1022", "1122", "0100", "2021",
         "0010", "2201", "2111", "0020",
         "1102", "0210", "2110", "1020"]) == 6, "failed on unique sets"
    
         
def test_is_set():
                                                                                    #test for unique cards
    with pytest.raises(ValueError) as excinfo:
        specs.is_set("1022", "1122", "1022")
    assert excinfo.value.args[0] == "the cards must all be unique"
                                                                                    #test for 4 digits
    with pytest.raises(ValueError) as excinfo:
        specs.is_set("1022", "122", "0100")
    assert excinfo.value.args[0] == "all cards must have exactly 4 digits"
                                                                                    #test for values 0,1,2
    with pytest.raises(ValueError) as excinfo:
        specs.is_set("1022", "1132", "0100")
    assert excinfo.value.args[0] == "all characters must be 0, 1, or 2"
                                                                                    #tests for all_equal and all_diff
    assert specs.is_set("1000", "1212", "1121") == True, "failed on True"
    assert specs.is_set("1100", "0100", "2222") == False, "failed on False"
