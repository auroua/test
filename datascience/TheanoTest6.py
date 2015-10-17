#encoding:UTF-8
__author__ = 'auroua'
import theano_test
from theano_test import pp
from theano_test import function
import theano_test.tensor as T
x = T.dscalar('x')
y = x**2
gy = T.grad(y,x)
f = function([x],y)

print f(4)


x2 = T.dmatrix('x2')
s = T.sum(1/(1+T.exp(-x2)))
gs = T.grad(s,x2)
dlogistic = function([x2],gs)
print dlogistic([[0,1],[-1,-2]])

x3 = T.dvector('x3')
y3 = x3**2
J,updates = theano_test.scan(lambda i,y,x:T.grad(y[i],x),sequences=T.arange(y3.shape[0]),non_sequences=[y3,x3])
f = function([x3],J,updates=updates)
print f([4,4])

x4 = T.dvector('x4')
y4 = x4**2
cost = y4.sum()
gy4 = T.grad(cost,x4)
H,updates2 = theano_test.scan(lambda i,gy,x4:T.grad(gy[i],x4),sequences=T.arange(gy4.shape[0]),non_sequences=[gy4,x4])
f2 = function([x4],H,updates=updates2)
print f2([4,4])

W = T.dmatrix('W')
V = T.dmatrix('V')
xx = T.dvector('xx')
yy = T.dot(xx,W)
JV = T.Rop(yy,W,V)
fwv = theano_test.function([W,V,xx],JV)
print fwv([[1, 1], [1, 1]], [[2, 2], [2, 2]], [0,1])
