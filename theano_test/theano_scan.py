import theano
import theano.tensor as T
import numpy as np

k = T.iscalar('k')
A = T.vector('A')

result, updates = theano.scan(fn=lambda prior_result, A: prior_result*A,
                              outputs_info=T.ones_like(A),
                              non_sequences=A, n_steps=k)
# final_result = result[-1]
power = theano.function(inputs=[A, k], outputs=result, updates=updates)

print power(range(10), 2)
print power(range(10), 4)

# coefficients = theano.tensor.vector('cofficients')
# x = T.scalar('x')
# max_cofficients_supported = 10000
#
# components, updates = theano.scan(fn=lambda cofficient, power, free_variable:
#                                       cofficient*(free_variable**power), outputs_info=None,
#                                       sequences=[coefficients, theano.tensor.arange(max_cofficients_supported)],
#                                       non_sequences=x)
#
# # sum them up
# polynomial = components.sum()
#
# # compile a function
# calculate_polynomial = theano.function(inputs=[coefficients, x], outputs=polynomial)
#
# # test
# test_cofficients = np.asarray([1, 0, 2], dtype=np.float32)
# test_value = 3
#
# print calculate_polynomial(test_cofficients, test_value)

# up_to = T.iscalar('up_to')
# def accumulate_by_adding(arange_val, sum_to_date):
#     return sum_to_date + arange_val
#
# # the initial output state that is supplied, that is outputs_info,
# # must be of a shape similar to that of the output variable
# # generated at each iteration and moreover
# seq = T.arange(up_to)
# # An unauthorized implicit downcast from the dtype of 'seq', to that of
# # 'T.as_tensor_variable(0)' which is of dtype 'int8' by default would occur
# # if this instruction were to be used instead of the next one:
# # outputs_info = T.as_tensor_variable(0)
# outputs_info = T.as_tensor_variable(np.asarray(0, seq.dtype))
# scan_result, scan_update = theano.scan(fn=accumulate_by_adding, outputs_info=outputs_info,
#                                        sequences=seq)
# triangular_sequence = theano.function(inputs=[up_to], outputs=scan_result)
# # test
# some_num = 15
# print(triangular_sequence(some_num))
# print([n * (n + 1) // 2 for n in xrange(some_num)])