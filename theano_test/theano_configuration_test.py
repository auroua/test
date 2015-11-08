__author__ = 'auroua'
import numpy as np
import theano
import theano.tensor as T

# N = 400
# feats = 784
# D = (np.random.randn(N, feats).astype(theano.config.floatX), np.random.randint(size=N, low=0, high=2).astype(theano.config.floatX))
# training_step = 10000
#
# x = T.dmatrix('x')
# y = T.dvector('y')
# w = theano.shared(np.random.randn(feats).astype(theano.config.floatX), name='w')
# b = theano.shared(np.asarray(0., dtype=theano.config.floatX), name='b')
# x.tag.test_value = D[0]
# y.tag.test_value = D[1]
#
# p_1 = 1./(1.+T.exp(-T.dot(x, w)-b))
# prediction = p_1 > 0.5
# x_ent = -y*T.log(p_1)-(1-y)*T.log(1-p_1)
# cost = x_ent.mean()+ 0.01*(w**2).sum()
# gw, gb = T.grad(cost, [w, b])
#
# train = theano.function(inputs=[x, y], outputs=[prediction, x_ent], updates=((w, w-0.01*gw), (b, b-0.01*gb)), name='train')
# predict = theano.function(inputs=[x], outputs=prediction, name='predict')
#
# if any([x.op.__class__.__name__ in ['Gemv', 'CGmv', 'Gemm', 'CGemm'] for x in
#         train.maker.fgraph.toposort()]):
#     print 'Used the cpu'
# elif any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for x in
#           train.maker.fgraph.toposort()]):
#     print 'Used the gpu'
# else:
#     print('ERROR, not able to tell if theano used the cpu or the gpu')
#     print(train.maker.fgraph.toposort())
#
# for i in range(training_step):
#     pred, err = train(D[0], D[1])
#
# print("target values for D")
# print(D[1])
#
# print("prediction on D")
# print(predict(D[0]))

# x = T.dvector('x')
# f = theano.function(inputs=[x], outputs=10*x, mode='DebugMode')
# f([5])
# f([0])
# f([7])

from theano import ProfileMode
profmode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())
v1, v2 = T.vectors(2)
o = v1 + v2
f = theano.function([v1,v2],[o], mode=profmode)