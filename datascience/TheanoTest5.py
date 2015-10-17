#encoding:UTF-8
__author__ = 'auroua'
import numpy
import theano_test
import theano_test.tensor as T

rng = numpy.random
N = 400
feats = 784
D = (rng.randn(N,feats).astype(theano_test.config.floatX),rng.randint(size=N,low=0,high=2).astype(theano_test.config.floatX))
traing_steps = 10000

x = T.dmatrix('x')
y = T.vector('y')
w = theano_test.shared(rng.rand(feats).astype(theano_test.config.floatX),name='w')
b = theano_test.shared(numpy.asarray(0.,dtype=theano_test.config.floatX),name='b')
x.tag.test_value = D[0]
y.tag.test_value = D[1]

p_1 = 1/(1+T.exp(-T.dot(x,w)-b))
prediction = p_1>0.5

xent = -y*T.log(p_1) - (1-y)*T.log(1-p_1)
cost = xent.mean() + 0.01*(w**2).sum()

gw,gb = T.grad(cost, [w,b])

train = theano_test.function(inputs=[x,y], outputs=[prediction, xent], updates=[[w, w-0.01*gw], [b, b-0.01*gb]], name = "train")
predict = theano_test.function(inputs=[x], outputs=prediction, name = "predict")

print theano_test.printing.pprint(prediction)
print theano_test.printing.debugprint(prediction)
print theano_test.printing.debugprint(predict)