__author__ = 'auroua'
import theano
import theano.tensor as T
import numpy as np

# example1
# x = T.matrix('x')
# w = T.matrix('w')
# b_sym = T.vector('b_sym')
#
# results, updates = theano.scan(lambda v: T.tanh(T.dot(v, w) + b_sym), sequences=x)
# compute_elementwise = theano.function(inputs=[x, w, b_sym], outputs=[results])
#
# x = np.eye(2, dtype=theano.config.floatX)
# w = np.ones((2, 2), dtype=theano.config.floatX)
# b = np.ones((2), dtype=theano.config.floatX)
# b[1] = 2
#
# print compute_elementwise(x, w, b)[0]
#
# print np.tanh(x.dot(w)+b)

# example2
# k = T.iscalar('k')
# a = T.vector('a')
#
# result, updates = theano.scan(fn=lambda prior_result, a:prior_result*a,outputs_info=T.ones_like(a), non_sequences=a, n_steps=k)
#
# final_result = result[-1]
# final_result = result
#
# power = theano.function(inputs=[a, k], outputs=final_result, updates=updates)
#
# print power(range(10), 2)
# print power(range(10), 4)

# example 3
# cofficients = T.vector('cofficients')
# x = T.scalars('x')
#
# max_cofficients_supported = 10000
#
# components, updates = theano.scan(fn=lambda cofficient, power, free_variable: cofficient*(free_variable**power),
#                                   outputs_info=None,
#                                   sequences=[cofficients, theano.tensor.arange(max_cofficients_supported)],
#                                   non_sequences=x)
#
# polynomial = components.sum()
#
# calculate_polynomial = theano.function(inputs=[cofficients, x], outputs=polynomial)
#
# test_cofficients = np.asarray([1, 0, 2], dtype=theano.config.floatX)
# test_value = 3
# print calculate_polynomial(test_cofficients, test_value)
# print 1.0*(3**0)+0.0*(3**1)+2.0*(3**2)

# example 4
# up_to = T.iscalar('up_to')
#
# def accumulate_by_adding(arange_val, sum_to_date):
#     return sum_to_date + arange_val
# seq = T.arange(up_to)
# print seq
# print seq.dtype
# outputs_info = T.as_tensor_variable(np.asarray(0, seq.dtype))
# scan_result, scan_updates = theano.scan(fn=accumulate_by_adding, outputs_info=outputs_info,
#                                         sequences=seq)
# triangular_sequence = theano.function(inputs=[up_to], outputs=scan_result)
#
# some_num = 15
# print triangular_sequence(some_num)
# print [n*(n+1)//2 for n in xrange(some_num)]


# example 5
# x =  T.vector('x')
# w = T.matrix('w')
# b_sym = T.vector('b_sym')
# u = T.matrix('u')
# y = T.matrix('y')
# v = T.matrix('v')
# p = T.matrix('p')
#
# results, updates = theano.scan(lambda y, p, x_tm1: T.tanh(T.dot(x_tm1, w)+T.dot(y, u)+ T.dot(p, v)),
#                                sequences=[y, p[::-1]], outputs_info=[x])
# compute_seq = theano.function(inputs=[x, w, y, u, p, v], outputs=[results])
#
# # test_values
# x = np.zeros((2), dtype=theano.config.floatX)
# x[1] = 1
# w = np.ones((2, 2), dtype=theano.config.floatX)
# y = np.ones((5, 2), dtype=theano.config.floatX)
# y[0, :] = -3
# u = np.ones((2, 2), dtype=theano.config.floatX)
# p = np.ones((5, 2), dtype=theano.config.floatX)
# p[0, :] = 3
# v = np.ones((2, 2), dtype=theano.config.floatX)
#
# print compute_seq(x, w, y, u, p, v)[0]

# compute one line
# define tensor variable
# X = T.matrix("X")
# results, updates = theano.scan(lambda x_i: T.sqrt((x_i ** 2).sum()), sequences=[X])
# compute_norm_lines = theano.function(inputs=[X], outputs=[results])
#
# # test value
# x = np.diag(np.arange(1, 6, dtype=theano.config.floatX), 1)
# print(compute_norm_lines(x)[0])
#
# # comparison with numpy
# print(np.sqrt((x ** 2).sum(1)))


# compute one column
# define tensor variable
# X = T.matrix("X")
# results, updates = theano.scan(lambda x_i: T.sqrt((x_i ** 2).sum()), sequences=[X.T])
# compute_norm_cols = theano.function(inputs=[X], outputs=[results])
#
# # test value
# x = np.diag(np.arange(1, 6, dtype=theano.config.floatX), 1)
# print(compute_norm_cols(x)[0])
#
# # comparison with numpy
# print(np.sqrt((x ** 2).sum(0)))


# compute the Jacobian
# v = T.vector()
# A = T.matrix()
# y = T.tanh(T.dot(v, A))
# results, updates = theano.scan(lambda i: T.grad(y[i], v), sequences=[T.arange(y.shape[0])])
# compute_jac_t = theano.function([A, v], [results], allow_input_downcast=True) # shape (d_out, d_in)
#
# # test values
# x = np.eye(5, dtype=theano.config.floatX)[0]
# w = np.eye(5, 3, dtype=theano.config.floatX)
# w[2] = np.ones((3), dtype=theano.config.floatX)
# print compute_jac_t(w, x)[0]