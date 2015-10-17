#encoding:UTF-8
__author__ = 'auroua'

import theano_test.tensor as T
from theano_test import function
import theano_test
import pydot

print pydot.find_graphviz()

x = T.dmatrix('x')
y = x*2

print type(y.owner)
print y.owner.op.name
print len(y.owner.inputs)
print type(y.owner.inputs[1].owner)

#apply nodes are those that define which computations the graph does
# When compiling a Theano function, what you give to the theano.function is actually a graph
# (starting from the output variables you can traverse the graph up to the input variables).
# While this graph structure shows how to compute the output from the input,
# it also offers the possibility to improve the way this computation is carried out.

a = T.vector('a')
b = a+a**10
fab = function([a],b)
print fab([0,1,2])

theano_test.printing.pydotprint(b, outfile="/home/auroua/symbolic_graph_unopt.png", var_with_name_simple=True)
theano_test.printing.pydotprint(fab, outfile="/home/auroua/symbolic_graph_opt.png", var_with_name_simple=True)
