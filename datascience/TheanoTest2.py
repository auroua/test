#encoding:UTF-8
__author__ = 'auroua'
import theano_test.tensor as T
import theano_test
from theano_test import function
x = T.dmatrix('x')
s = 1/(1+T.exp(-x))
logistic = function([x],s)
print logistic([[0,1],[-1,-2]])

s2 = (1+T.tanh(x/2))/2
logistic2 = function([x],s2)
print logistic2([[0,1],[-1,-2]])

a,b = T.matrices('x','y')
diff = a-b
abs_diff = abs(diff)
diff_squared = diff**2
f = function([a,b],[diff,abs_diff,diff_squared])

print f([[1, 1], [1, 1]], [[0, 1], [2, 3]])

x,y,w = T.dscalars('x','y','w')
z = (x+y)*w
f2 = function([x,theano_test.Param(y,default=1),theano_test.Param(w,default=2,name='w_by_name')],z)
print f2(33)
print f2(33,0,1)
print f2(33,w_by_name=1)
print f2(33,w_by_name=1,y=0)

print '++++++++++++++++++++++++++++++++++++'

from theano_test import shared
state = shared(0)
inc = T.iscalar('inc')
accumlator = function([inc],state,updates=[(state,state+inc)])
decrementor = function([inc],state,updates=[(state,state-inc)])

#每调用一次accumlator方法,共享变量state会自增一次,不调用不自增,所有对accumlator方法的调用都是共享同一个state变量
print state.get_value()
print accumlator(1)
print state.get_value()
print accumlator(300)
print state.get_value()
state.set_value(-1)
print accumlator(3)
print state.get_value()
decrementor(2)
print state.get_value()

# In practice, a good way of thinking about the givens is as a mechanism that allows you to
# replace any part of your formula with a different expression that evaluates to a tensor of same shape and dtype.
fn_of_state = state*2+inc
foo = T.scalar(dtype=state.dtype)
skip_shared = function([inc,foo],fn_of_state,givens=[(state,foo)])
print skip_shared(1,3)
print state.get_value()

from theano_test.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=234)
rv_u = srng.uniform((2,2))
rv_n = srng.normal((2,2))
f = function([],rv_u)
g = function([],rv_n,no_default_updates=True)
#An important remark is that a random variable is drawn
# at most once during any single function execution.
nearly_zeros = function([],rv_u+rv_u-2*rv_u)
print f()
print f()
print g()
print g()
print nearly_zeros()

rng_val = rv_u.rng.get_value(borrow=True)
rng_val.seed(89234)
rv_u.rng.set_value(rng_val,borrow=True)
srng.seed(902340)

#还原到之前的某个随机状态
state_after_v0 = rv_u.rng.get_value().get_state()
nearly_zeros()
v1=f()
rng = rv_u.rng.get_value(borrow=True)
rng.set_state(state_after_v0)
rv_u.rng.set_value(rng,borrow=True)
v2=f()
v3=f()
print '====================================='
print v1
print v2
print v3

print '====================================='
print '====================================='
class Graph(object):
    def __init__(self,seed=123):
        self.rng = RandomStreams(seed);
        self.y = self.rng.uniform(size=(1,))

g1 = Graph(seed=123)
f1 = theano_test.function([],g1.y)

g2 = Graph(seed=987)
f2 = theano_test.function([],g2.y)

print 'By default, the two functions are out of sync.'
print 'f1() returns ', f1()
print 'f2() returns ', f2()

from theano_test.sandbox.rng_mrg import MRG_RandomStreams
def copy_random_state(g1,g2):
    if isinstance(g1.rng,MRG_RandomStreams):
        g2.rng.rstate = g1.rng.rstate
    for (su1,su2) in zip(g1.rng.state_updates,g2.rng.state_updates):
        su2[0].set_value(su1[0].get_value())

print 'We now copy the state of the theano random number generators.'
copy_random_state(g1, g2)
print 'f1() returns ', f1()
print 'f2() returns ', f2()