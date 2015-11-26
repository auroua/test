#encoding:UTF-8
__author__ = 'auroua'
import numpy
import theano
import theano.tensor as T
import cPickle
rng = numpy.random

N = 400
feats = 784
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
print D[0].shape, D[1].shape
training_steps = 10000

# Declare Theano symbolic variables
x = T.matrix('x')
y = T.vector('y')
w = theano.shared(rng.randn(feats), name='w')
b = theano.shared(0., name='b')

# print 'Initial model:'
# print w.get_value()
# print b.get_value()

p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))         # Probability that target = 1
prediction = p_1 > 0.5                          # The prediction thresholded
x_ent = -y * T.log(p_1) - (1-y)*(T.log(1-p_1))  # Cross-entropy loss function
cost = x_ent.mean() + 0.01*(w**2).sum()         # The cost to minimize     0.01*(w**2).sum() 正则化项
gw, gb = T.grad(cost, [w, b])                   # Compute the gradient of the cost we shall return to this in a following section of this tutorial

# compile
train = theano.function(inputs=[x, y], outputs=[prediction, x_ent],
                        updates=((w, w-0.1*gw), (b, b-0.1*gb)))
predict = theano.function(inputs=[x], outputs=prediction)

for i in range(training_steps):
    pred, err = train(D[0], D[1])
    print err.sum()

f = file('/home/aurora/hdd/workspace/PycharmProjects/data/rbm_weights.save', 'wb')
for obj in [w, b]:
    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

# print 'Final model:'
# print w.get_value()
# print b.get_value()
# print 'target values for D:'
# print D[1]
# print 'prediction on D:'
# print predict(D[0])


f = file('/home/aurora/hdd/workspace/PycharmProjects/data/rbm_weights.save', 'rb')
w = cPickle.load(f)
b = cPickle.load(f)
f.close()

print type(w)
print w.get_value()
