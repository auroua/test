__author__ = 'auroua'
from theano import *
import theano.tensor as T
import numpy as np
import cPickle,gzip

x = T.dvector('x')
y = T.dvector('y')
A = T.dmatrix('A')

z = x+A.dot(y)

f = function([x,y,A],z)

x_data = np.random.random(10)
y_data = np.random.random(5)
A_data = np.random.random((10,5))

print f(x_data,y_data,A_data)
