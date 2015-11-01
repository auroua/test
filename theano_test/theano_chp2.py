__author__ = 'auroua'

import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams

if __name__=='__main__':
    # logistic function
    # data = np.linspace(-5, 5, 1000)
    # print data
    # data_y = 1/(1+np.exp(data*-1))
    # ax = plt.subplot(1, 1, 1)
    #
    # ax.scatter(data, data_y)
    # plt.grid()
    # plt.show()

    # example1
    # x = T.dmatrix('x')
    # s = 1 / (1 + T.exp(-x))
    # logistic = theano.function([x], s)
    # print logistic([[0, 1], [-1, -2]])
    #
    # s2 = (1 + T.tanh(x / 2)) / 2
    # logistic2 = theano.function([x], s2)
    # print logistic2([[0, 1], [-1, -2]])

    # example 2
    # a, b = T.dmatrices('a', 'b')
    # diff = a - b
    # abs_diff = abs(diff)
    # diff_squared = diff**2
    # f = theano.function([a, b], [diff, abs_diff, diff_squared])
    # print f([[1, 1], [1, 1]],[[0, 1], [2, 3]])

    # example 3
    # x, y = T.dscalars('x', 'y')
    # z = x + y
    # f = theano.function([x, theano.Param(y, default=1)], z)
    # print f(33)

    # x, y, w = T.dscalars('x', 'y', 'w')
    # z = (x + y) * w
    # f = theano.function([x, theano.Param(y, default=1), theano.Param(w, default=2, name='w_by_name')], z)
    # print f(33), f(33, 2), f(33, 0, 1), f(33, w_by_name=1), f(33, w_by_name=1, y=0)

    # example 4
    # state = theano.shared(0)
    # inc = T.iscalar('inc')
    # accumulator = theano.function([inc], state, updates=[(state, state+inc)])
    # decrementor = theano.function([inc], state, updates=[(state, state-inc)])
    #
    # print state.get_value()
    # # return the old state value
    # print accumulator(1)
    # print state.get_value()
    # print accumulator(300)
    # print state.get_value()
    #
    # state.set_value(-1)
    # print accumulator(3)
    # print state.get_value()
    #
    # print decrementor(2)
    # print state.get_value()
    #
    # fn_of_state = state*2 + inc
    # # return the data type of state the data type is not shared variable but int64
    # # The type of foo must match the shared variable we are replacing
    # # with the ``givens``
    # foo = T.scalar(dtype=state.dtype)
    # # we're using 3 for the state, not state.value
    # skip_shared = theano.function([inc, foo], fn_of_state, givens=[(state, foo)])
    # print skip_shared(1, 3)
    # # old state still there, but we didn't use it
    # state.get_value()

    # example 5
    # srng = RandomStreams(seed=234)
    # rv_u = srng.uniform((2, 2))
    # rv_n = srng.normal((2, 2))
    # f = theano.function([], rv_u)
    # g = theano.function([], rv_n, no_default_updates=True) #Not updating rv_n.rng
    # # a random variable is drawn at most once during any single function execution
    # nearly_zeros = theano.function([], rv_u + rv_u - 2*rv_u)
    #
    # print f()
    # print f()
    # print g()
    # print g()
    #
    # print nearly_zeros()
    #
    # rng_val = rv_u.rng.get_value(borrow=True)
    # rng_val.seed(89234)
    # rv_u.rng.set_value(rng_val, borrow=True)
    #
    # state_after_v0 = rv_u.rng.get_value().get_state()
    # print nearly_zeros()
    # v1 = f()
    # print v1
    # rng = rv_u.rng.get_value(borrow=True)
    # rng.set_state(state_after_v0)
    # rv_u.rng.set_value(rng, borrow=True)
    # v2 = f()
    # v3 = f()
    # print v2, v3

    # example 6
    class Graph():
        def __init__(self, seed=123):
            self.rng = RandomStreams(seed)
            self.y =self.rng.uniform(size=(1,))

    g1 = Graph(seed=123)
    f1 = theano.function([], g1.y)

    g2 = Graph(seed=987)
    f2 = theano.function([], g2.y)

    print f1()
    print f2()

    def copy_random_state(g1, g2):
        if isinstance(g1.rng, MRG_RandomStreams):
            g2.rng.rstate = g1.rng.rstate
        for (su1, su2) in zip(g1.rng.state_updates, g2.rng.state_updates):
            su2[0].set_value(su1[0].get_value())

    copy_random_state(g1, g2)
    print f1(), f2()