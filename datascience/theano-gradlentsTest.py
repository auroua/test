#encoding:UTF-8
__author__ = 'auroua'
from theano_test import pp
import theano_test.tensor as T
from theano_test import function

#简单标量函数的求导
x = T.dscalar('x')
y = x ** 2
gy = T.grad(y,x)
print pp(gy)
f = function([x],gy)
print f(4)
print f(94.2)
print pp(f.maker.fgraph.outputs[0])

#sigmodi的求导
xs = T.dmatrix('x')
y = T.sum(1/(1+T.exp(-xs)))
gs = T.grad(y,xs)
dlogistic = function([xs],gs)
print dlogistic([[0, 1], [-1, -2]])