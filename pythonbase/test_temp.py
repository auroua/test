__author__ = 'auroua'
import nose

from pythonbase import temp


def test_to_celsius():
    '''test to_celsius method'''
    assert temp.to_celsius(32)==0

def test_boiling():
    '''Test boiling point.'''
    assert temp.to_celsius(212)==100

def test_distance():
    assert temp.distance(0,3,4,3)==4,'Test distances'

if __name__=='__main__':
    nose.runmodule()