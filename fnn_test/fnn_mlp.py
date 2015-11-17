# encoding:UTF-8
__author__ = 'auroua'
import theano
import theano.tensor as T
import numpy as np
import gzip, cPickle
import matplotlib.pyplot as plt

def encode_labels(label, length):
    encode_label = np.zeros((label.shape[0], length+1))
    for i in range(label.shape[0]):
        encode_label[i, label[i]] = 1
    return encode_label

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

with gzip.open('/home/aurora/workspace/PycharmProjects/data/MNIST/mnist.pkl.gz','rb') as f:
    train_set, validate_set, test_set = cPickle.load(f)

w1_shape = (50, 784)
b1_shape = 50
w2_shape = (10, 50)
b2_shape = 10

w1 = theano.shared(np.random.randn(*w1_shape), name='w1')
b1 = theano.shared(np.random.randn(b1_shape), name='b1')
w2 = theano.shared(np.random.randn(*w2_shape), name='w2')
b2 = theano.shared(np.random.randn(b2_shape), name='b2')

x = T.dmatrix('x')
labels = T.dmatrix('labels')

hidden = T.nnet.sigmoid(x.dot(w1.T)+b1)
output = T.nnet.softmax(hidden.dot(w2.T)+b2)
prediction = T.argmax(output, axis=1)
reg_lambda = 0.0001
regularization = reg_lambda*((w1**2).sum()+(w2**2).sum()+(b1**2).sum()+(b2**2).sum())
cost = T.nnet.binary_crossentropy(output, labels).mean() + regularization

computer_prediction = theano.function([x], prediction)

alpha = T.dscalar('alpha')
weights = [w1, w2, b1, b2]
updates = [(w, w - alpha*T.grad(cost, w)) for w in weights]
train_nn = theano.function(inputs=[x, labels, alpha], outputs=[hidden, cost], updates=updates)

alpha_val = 10.0
labeled = encode_labels(train_set[1], 9)

costs = []
hidden_val = []
while True:
    hid_val, cost = train_nn(train_set[0], labeled, alpha_val)
    costs.append(float(cost))
    hidden_val.append(hidden_val)

    if len(costs) % 10 == 0:
        print 'Epoch', len(costs), 'with cost', costs[-1], 'and alpha', alpha_val
    if len(costs) > 2 and costs[-2] - costs[-1] < 0.0001:
        if alpha_val < 0.2:
            print 'Epoch', len(costs), 'with cost', costs[-1], 'and alpha', alpha_val
            break
        else:
            alpha_val /= 1.5

predict = computer_prediction(test_set[0])
print accuracy(predict, test_set[1])

val_w1 = w1.get_value()
activations = [val_w1[i, :].reshape((28, 28)) for i in range(val_w1.shape[0])]

for index, w in enumerate(activations):
    plt.subplot(5, 10, index+1)
    plt.set_cmap('gray')
    plt.axis('off')
    plt.imshow(w)
plt.subplots_adjust(hspace=-0.85)
plt.gcf().set_size_inches(9,9)
plt.show()

plt.subplot(2, 3, 1)
n, bins, patches = plt.hist(b1.get_value(), 20, normed=1, histtype='stepfilled')
plt.title('Bias b1')
plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)

plt.subplot(2, 3, 2)
n, bins, patches = plt.hist(b2.get_value(), 20, normed=1, histtype='stepfilled')
plt.title('Bias b2')
plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)

plt.subplot(2, 3, 3)
n, bins, patches = plt.hist(w1.get_value().flatten(), 50, normed=1, histtype='stepfilled')
plt.title('Weights w1')

plt.subplot(2, 3, 4)
n, bins, patches = plt.hist(w2.get_value().flatten(), 50, normed=1, histtype='stepfilled')
plt.title('Weights w2')

plt.subplot(2, 3, 5)
plt.plot(range(len(costs)), costs)
plt.title('cost vs training epoch')

plt.gcf().set_size_inches(10, 5)
plt.show()