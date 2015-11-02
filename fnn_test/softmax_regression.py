__author__ = 'auroua'
from theano import *
import theano.tensor as T
import numpy as np
import cPickle, gzip
import matplotlib.pyplot as plt

# x = T.dvector('x')
# y = T.dvector('y')
# A = T.dmatrix('A')
#
# z = x+A.dot(y)
#
# f = function([x,y,A],z)
#
# x_data = np.random.random(10)
# y_data = np.random.random(5)
# A_data = np.random.random((10,5))
#
# print f(x_data,y_data,A_data)

def encode_labels(labels, max_index):
    '''Encode the labels into binary vectors.'''
    # Allocate the output labels, all zeros.
    encoded = np.zeros((labels.shape[0], max_index+1))
    for i in xrange(labels.shape[0]):
        encoded[i, labels[i]] = 1
    return encoded

with gzip.open('/home/auroua/workspace/PycharmProjects/data/MNIST/mnist.pkl.gz','rb') as f:
    train_set, valid_set, test_set = cPickle.load(f)

# print train_set[0].shape,train_set[1].shape
# print valid_set[0].shape,valid_set[1].shape
# print test_set[0].shape,test_set[1].shape

w_shape = (10, 784)
b_shape = 10

w = theano.shared(np.random.randn(*w_shape), name='w')
b = theano.shared(np.random.randn(b_shape), name='b')

x = T.dmatrix('x')
labels = T.dmatrix('labels')

# w:10*784   output : 50000*10
output = T.nnet.softmax(x.dot(w.T)+b)
prediction = T.argmax(output, axis=1)    # prediction: 50000*1
# cost = T.nnet.binary_crossentropy(output, labels).mean()
# cost = T.nnet.binary_crossentropy(output, labels).mean() + 0.01*(w**2).sum()  #85.xx
cost = T.nnet.binary_crossentropy(output, labels).mean() + 0.001*(w**2).sum()  #90.39

compute_prediction = theano.function([x], prediction)
compute_cost = theano.function([x, labels], cost)
grad_w = theano.grad(cost, w)
grad_b = theano.grad(cost, b)
alpha = T.dscalar('alpha')
train_fun = theano.function([x, labels, alpha], cost, updates=((w, w-alpha*grad_w), (b, b-alpha*grad_b)))

alpha_val = 10.0
labeled = encode_labels(train_set[1], 9)
costs = []
while True:
    costs.append(float(train_fun(train_set[0], labeled, alpha_val)))

    if len(costs)%10 == 0:
        print 'Epoch', len(costs), 'with cost', costs[-1], 'and alpha', alpha_val
    if len(costs) > 2 and costs[-2]-costs[-1] < 0.0001:
        if alpha_val < 0.2:
            break
        else:
            alpha_val /= 1.5

predict = compute_prediction(test_set[0])

def accuracy(predicted, actual):
    total = 0.0
    correct = 0.0
    # correct = (predicted==actual).sum()
    # return correct/predicted[0]

    for p, a in zip(predicted, actual):
        total += 1
        if p == a:
            correct += 1

    return correct/total

print accuracy(predict, test_set[1])

val_w = w.get_value()
activations = [val_w[i, :].reshape((28, 28)) for i in xrange(val_w.shape[0])]

for i, w in enumerate(activations):
    plt.subplot(1, 10, i+1)
    plt.set_cmap('gray')
    plt.axis('off')
    plt.imshow(w)
plt.gcf().set_size_inches(9, 9)
plt.show()