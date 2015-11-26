import rbm_theano as crbm
import theano.tensor as T
import theano
import numpy as np
import gzip, cPickle
from theano.tensor.shared_randomstreams import RandomStreams

theano.config.compute_test_value='raise'

with gzip.open('/home/aurora/workspace/PycharmProjects/data/MNIST/mnist.pkl.gz', 'rb') as f:
        train_set, validate_set, test_set = cPickle.load(f)
x = T.dmatrix('x')
train_set_x, train_set_y = train_set
print train_set_x[:20].shape
x.tag.test_value = train_set_x[:20]

rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2**30))
rbm = crbm.RBM(input=x, n_visible=28*28, n_hidden=500, numpy_rng=rng, theano_rng=theano_rng)
persistent_chain = theano.shared(np.zeros((20, 500), dtype=theano.config.floatX), borrow=True)
chain_start = persistent_chain
print chain_start.get_value()
(
    [
        pre_sigmoid_nvs,
        nv_means,
        nv_samples,
        pre_sigmoid_nhs,
        nh_means,
        nh_samples
    ],
    updates
) = theano.scan(fn=rbm.gibbs_hvh, outputs_info=[None, None, None, None, None, chain_start], n_steps=15)

# pre_sigmoid, sigmoid, samples = rbm.sample_h_given_v(x)
# print pre_sigmoid.tag.test_value.shape
# print sigmoid.tag.test_value.shape
# print samples.tag.test_value.shape

samplesing = theano.function([], pre_sigmoid_nvs, mode='DebugMode', updates=updates)
results = samplesing()
print results.shape
print results[-1]
print updates