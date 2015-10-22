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



with gzip.open('/home/auroua/workspace/PycharmProjects/data/MNIST/mnist.pkl.gz','rb') as f:
    train_set,valid_set,test_set = cPickle.load(f)

print train_set[0].shape,train_set[1].shape
print valid_set[0].shape,valid_set[1].shape
print test_set[0].shape,test_set[1].shape

print train_set[0]