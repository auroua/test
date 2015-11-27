from theano.tensor.signal import downsample
import theano.tensor as T
import theano
import numpy as np

img_input = T.dtensor4('input')
maxpool_shape = (2, 2)
pool_out = downsample.max_pool_2d(img_input, maxpool_shape, ignore_border=True)
f = theano.function([img_input], pool_out)

invals = np.random.RandomState(1).rand(3, 2, 5, 5)
print 'with ignore_border set to True:'
print 'invals[0, 0, :, :]=\n', invals[0, 0, :, :]
print 'output[0, 0, :, :]=\n', f(invals)[0, 0, :, :]

pool_out = downsample.max_pool_2d(img_input, maxpool_shape, ignore_border=False)
f = theano.function([img_input], pool_out)
print 'With ignore_border set to False:'
print 'invals[1, 0, :, :]=\n ', invals[0, 0, :, :]
print 'output[1, 0, :, :]=\n ', f(invals)[0, 0, :, :]
