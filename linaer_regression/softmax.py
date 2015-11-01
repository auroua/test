__author__ = 'auroua'

import cPickle,gzip
import numpy as np
from theano import shared
import theano.tensor as t

if __name__ == '__main__':
    #image size 784 : 28*28
    with gzip.open('/home/auroua/workspace/PycharmProjects/data/MNIST/mnist.pkl.gz','rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)

    w_shape = (10,784)
    b_shape = (10)

    w = shared(np.random.random(w_shape)-0.5,name='w')
    b = shared(np.random.random(b_shape)-0.5,name='b')

    x = t.dmatrix('x') # n*784
    labels = t.dmatrix('labels')  #n*10

