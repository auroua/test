__author__ = 'auroua'
import theano.tensor as T
from theano import function,pp
x = T.dscalar('x')
y = T.dscalar('y')

z = x + y
f = function([x,y],z)

print f(2,3)
print pp(z)

print z.eval({x:16.3,y:12.3})
print z

xm = T.dmatrix('x')
ym = T.dmatrix('y')
zm = xm + ym

fm = function([xm,ym],zm)
print fm([[1,2],[3,4]],[[10,20],[30,40]])

exa = T.vector('exa')
exb = T.vector('exb')
exz = exa**2+exb**2+2*exa*exb

fex = function([exa,exb],exz)

print fex([1,2],[3,4])