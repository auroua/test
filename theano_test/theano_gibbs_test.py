import theano
from theano import tensor as T
import numpy as np

W_values = np.random.normal(size=(3, 5))
bvis_values = np.random.normal(size=(1, 3))
bhid_values = np.random.normal(size=(1, 5))

W = theano.shared(W_values) # we assume that ``W_values`` contains the
                            # initial values of your weight matrix
trng = T.shared_randomstreams.RandomStreams(1234)
def OneStep(vsample) :
    hmean = T.nnet.sigmoid(theano.dot(vsample, W) + bhid_values)
    hsample = trng.binomial(size=hmean.shape, n=1, p=hmean)
    vmean = T.nnet.sigmoid(theano.dot(hsample, W.T) + bvis_values)
    return trng.binomial(size=vsample.shape, n=1, p=vmean, dtype=theano.config.floatX)

sample = theano.tensor.matrix('sample')

values, updates = theano.scan(OneStep, outputs_info=sample, n_steps=10)

gibbs10 = theano.function([sample], values[-1], updates=updates)

samp = np.random.normal(size=(10, 3))
result3 = gibbs10(samp)
print result3

# a = theano.shared(1)
# values, updates = theano.scan(lambda: {a: a+1}, n_steps=10)
#
# b = a + 1
# c = updates[a] + 1
# f = theano.function([], [b, c], updates=updates)
#
# f()
# print b
# print(c)
# print(a.get_value())

# def v1a():
#     a = theano.shared(1)
#     outputs, updates = theano.scan(lambda x: x + 1, outputs_info=a, n_steps=10)
#     f = theano.function([], outputs=outputs)
#     print f(), a.get_value()
#
#
# def v1b():
#     a = theano.shared(1)
#     outputs, updates = theano.scan(lambda x: x + 1, outputs_info=a, n_steps=10)
#     f = theano.function([], outputs=outputs, updates=updates)
#     print f(), a.get_value()
#
#
# def v2a():
#     a = theano.shared(1)
#     outputs, updates = theano.scan(lambda: {a: a + 1}, n_steps=10)
#     f = theano.function([], outputs=outputs)
#     print f(), a.get_value()
#
#
# def v2b():
#     a = theano.shared(1)
#     outputs, updates = theano.scan(lambda: {a: a + 1}, n_steps=10)
#     f = theano.function([], outputs=outputs, updates=updates)
#     print f(), a.get_value()
#
#
# def main():
#     v1a()
#     v1b()
#     v2a()
#     v2b()
#
#
# main()