# from theano import config
# from theano import tensor as T
# config.compute_test_value = 'raise'
# import numpy as np
# #define a variable, and use the 'tag.test_value' option:
# x = T.matrix('x')
# x.tag.test_value = np.random.randint(100,size=(5,5))
#
# #define how y is dependent on x:
# y = x*x
#
# #define how some other value (here 'errorCount') depends on y:
# errorCount = T.sum(y)
#
# #print the tag.test_value result for debug purposes!
# print errorCount.tag.test_value
# print y.tag.test_value

# import theano
# import numpy
# x = theano.tensor.dvector('x')
#
# x_printed = theano.printing.Print('this is a very important value')(x)
#
# f = theano.function([x], x * 5)
# f_with_print = theano.function([x], x_printed * 5)
#
# #this runs the graph without any printing
# assert numpy.all( f([1, 2, 3]) == [5, 10, 15])
#
# #this runs the graph with the message, and value printed
# assert numpy.all( f_with_print([1, 2, 3]) == [5, 10, 15])


import theano


def inspect_inputs(i, node, fn):
    print i, node, "input(s) value(s):", [input[0] for input in fn.inputs],


def inspect_outputs(i, node, fn):
    print "output(s) value(s):", [output[0] for output in fn.outputs]

x = theano.tensor.dscalar('x')
f = theano.function([x], [5 * x], mode=theano.compile.MonitorMode(pre_func=inspect_inputs, post_func=inspect_outputs))
f(3)

# The code will print the following:
#   0 Elemwise{mul,no_inplace}(TensorConstant{5.0}, x) input(s) value(s): [array(5.0), array(3.0)] output(s) value(s): [array(15.0)]