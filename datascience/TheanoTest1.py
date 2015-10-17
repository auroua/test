__author__ = 'auroua'
import theano_test.tensor as T
from theano_test import function
from theano_test import pp
x = T.dscalar('x')
y = T.dscalar('y')
z = x+y
f = function([x,y],z)
f(2,3)
z.eval({x:16.3,y:14.3})
print z
print pp(z)

xm = T.dmatrix('xm')
ym = T.dmatrix('ym')
zm = xm+ym
f2 = function((xm,ym),zm)

f2(np.array([[1,2],[2,3]]),np.array([[3,4],[4,5]]))

xv = T.dvector('xv')
yv = T.dvector('yv')
zv = xv**2+yv**2+2*xv*yv
fv = function((xv,yv),zv)
print pp(zv)
print fv([1,2],[3,4])