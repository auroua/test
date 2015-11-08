__author__ = 'auroua'
import theano
import numpy as np
import theano.tensor as T
from theano import pp

# emample1
# x = T.dscalar('x')
# y = x**2
# gy = T.grad(y, x)
# print pp(gy)
#
# f = theano.function([x], gy)
# print f(4)
# print f(94.2)

# example 2
# In general, for any scalar expression s, T.grad(s, w) provides the Theano expression for computing
# x = T.dvector('x')
# w = theano.shared(np.random.randn(3, 3), 'w')
# b = theano.shared(np.random.randn(3), 'b')
# y = T.sum(1./(1 + T.exp(-x.dot(w.T)-b)))
#
# gw, gb = T.grad(y, [w, b])
#
# dlogistic = theano.function(inputs=[x], outputs=[gw, gb])
#
# gww, gbb = dlogistic([1.0, 1.1, 1.3])
#
# print gww
# print gbb

# example 3   Jacobian
# x = T.dvector('x')
# y = x ** 2
# J, updates = theano.scan(lambda  i, y, x:T.grad(y[i], x), sequences=T.arange(y.shape[0]), non_sequences=[y, x])
# f = theano.function([x], J, updates=updates)
# print f([4, 4])

# example 4 Hessian
x = T.dvector('x')
y = x**2
cost = y.sum()
gy = T.grad(cost, x)
H, updates = theano.scan(lambda i, gy, x: T.grad(gy[i], x), sequences=T.arange(gy.shape[0]), non_sequences=[gy, x])
f = theano.function([x], H, updates=updates)
print f([4, 4])




