__author__ = 'auroua'
import math

'Function for working with temperatures.'


def to_celsius(t):
    '''Convert Fahrenheit to Celsius.'''
    return round((t - 32.0) * 5.0 / 9.0)


def above_freezing(t):
    '''True if temperature in Celsius is above freezing, False others.'''
    return t > 0

def distance(x0,y0,x1,y1):
    '''caculate the Eduia distance'''
    return math.sqrt((x0-x1)**2+(y0-y1)**2)

if __name__ == 'temp':
    pass
    #print help(__name__)
