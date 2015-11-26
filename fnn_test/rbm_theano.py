"""This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.
"""
import timeit
import PIL.Image as Image
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import os
import gzip
import cPickle


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0/(ndar.max()+eps)
    return ndar


def tile_raster_image(x, img_shape, tile_shape, tile_spacing=(0, 0), scale_rows_to_unit_interval=True,
                      output_pixel_vals=True):
    """
        Transform an array with one flattened image per row, into an array in
        which images are reshaped and layed out like tiles on a floor.

        This function is useful for visualizing datasets whose rows are images,
        and also columns of matrices for transforming those rows
        (such as the first layer of a neural net).

        :type x: a 2-D ndarray or a tuple of 4 channels, elements of which can
        be 2-D ndarrays or None;
        :param x: a 2-D array in which every row is a flattened image.

        :type img_shape: tuple; (height, width)
        :param img_shape: the original shape of each image

        :type tile_shape: tuple; (rows, cols)
        :param tile_shape: the number of images to tile (rows, cols)

        :type tile_spacing: tuple; (rows, cols)
        :param tile_spacing: the number of images to tile (rows, cols)

        :param output_pixel_vals: if output should be pixel values (i.e. int8
        values) or floats

        :param scale_rows_to_unit_interval: if the values need to be scaled before
        being plotted to [0,1] or not

        :returns: array suitable for viewing as an image.
        (See:`Image.fromarray`.)
        :rtype: a 2-d array with same dtype as X.

    """
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape = [0,0]
    # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp)*tshp - tsp for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(x, tuple):
        assert len(x) == 4
        # create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='unit8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=x.dtype)
        # colors default to 0, alpha defaults to 1 (poaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if x[i] is None:
                # if channel is none, fill it with zeros of the correct dtype
                out_array[:, :, i] = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else out_array.dtype)+channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it in the output
                out_array[:, :, i] = tile_raster_image(x[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        h, wt = img_shape
        hs, ws = tile_spacing

        # generate a matrix to store the output
        out_array = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else x.dtype)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row*tile_shape[1]+tile_col<x.shape[0]:
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(x[tile_row*tile_shape[1]+tile_col].reshape(img_shape))
                    else:
                        this_img = x[tile_row*tile_shape[1]+tile_col].reshape(img_shape)
                    # add the slice to the corresponding position in the output array
                    out_array[tile_row*(h+hs): tile_row*(h+hs)+h, tile_col*(wt+ws): tile_col*(wt+ws)+wt] = this_img*(255 if output_pixel_vals else 1)
        return out_array


class RBM(object):
    """Restricted Boltzmann Machine(RBM)   """
    def __init__(self, input=None, n_visible=784, n_hidden=500, w=None,
                  hbias=None, vbias=None, numpy_rng=None, theano_rng=None):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param w: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            # create a number generator
            numpy_rng = np.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2**30))

        if w is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_w = np.asarray(
                numpy_rng.uniform(
                    low = -4*np.sqrt(6./n_visible),
                    high= 4*np.sqrt(6./n_hidden),
                    size= (n_visible, n_hidden)
                ), dtype=theano.config.floatX
            )
            # theano shared variables for weights and biases
            w = theano.shared(value=initial_w, name='w', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(
                value=np.zeros(n_hidden, dtype=theano.config.floatX),
                name='hbias', borrow=True
            )

        if vbias is None:
            vbias = theano.shared(
                value=np.zeros(n_visible, dtype=theano.config.floatX),
                name='vbias', borrow=True
            )

        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.w = w
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.w, self.hbias, self.vbias]

    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(vis, self.w) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(hid, self.w.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def free_energy(self, v_sample):
            ''' Function to compute the free energy
            the formula from the practical guide to training RBM'''
            wx_b = T.dot(v_sample, self.w) + self.hbias
            vbias_term = T.dot(v_sample, self.vbias)
            hidden_term = T.sum(T.log(1+T.exp(wx_b)), axis=1)
            return -hidden_term-vbias_term

    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
            """
            This functions implements one step of CD-k or PCD-k

            :param lr: learning rate used to train the RBM

            :param persistent: None for CD. For PCD, shared variable
                containing old state of Gibbs chain. This must be a shared
                variable of size (batch size, number of hidden units).

            :param k: number of Gibbs steps to do in CD-k/PCD-k

            Returns a proxy for the cost and the updates dictionary. The
            dictionary contains the update rules for weights and biases but
            also an update of the shared variable used to store the persistent
            chain, if one is used.

            """
            # compute positive phase
            pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

            # decide how to initialize persistent chain:
            # for CD, we use the newly generate hidden sample
            # for PCD, we initialize from the old state of the chain

            if persistent is None:
                chain_start = ph_sample
            else:
                chain_start = persistent

            # perform actual negative phase
            # in order to implement CD-k/PCD-k we need to scan over the
            # function that implements one gibbs step k times.
            # Read Theano tutorial on scan for more information :
            # http://deeplearning.net/software/theano/library/scan.html
            # the scan will return the entire Gibbs chain

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
            ) = theano.scan(
                    fn=self.gibbs_hvh,
                    # the None are place holders, saying that
                    # chain_start is the initial state corresponding to the
                    # 6th output
                    outputs_info=[None, None, None, None, None, chain_start],
                    n_steps=k
            )

            # determine gradients on RBM parameters
            # note that we only need the sample at the end of the chain
            chain_end = nv_samples[-1]
            cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))
            # We must not compute the gradient through the gibbs sampling
            gparams = T.grad(cost, self.params, consider_constant=[chain_end])

            for gparam, param in zip(gparams, self.params):
                updates[param] = param-gparam*T.cast(lr, dtype=theano.config.floatX)

            if persistent:
                # Note that this works only if persistent is a shared variable
                updates[persistent] = nh_samples[-1]
                # pseudo-likelihood is a better proxy for PCD
                monitoring_cost = self.get_pseudo_likelihood_cost(updates)
            else:
                # reconstruction cross-entropy is a better proxy for CD
                monitoring_cost = self.get_reconstruction_cost(updates, pre_sigmoid_nvs[-1])
            return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
            """Stochastic approximation to the pseudo-likelihood"""

            # index of bit i in expression p(x_i | x_{\i})
            bit_i_idx = theano.shared(value=0, name='bit_i_idx')

            # binarize the input image by rounding to nearest integer
            xi = T.round(self.input)

            # calculate free energy for the given bit configuration
            fe_xi = self.free_energy(xi)

            # flip bit x_i of matrix xi and preserve all other bits x_{\i}
            # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
            # the result to xi_flip, instead of working in place on xi.
            xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

            # calculate free energy with bit flipped
            fe_xi_flip = self.free_energy(xi_flip)

            # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
            cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))

            # increment bit_i_idx % number as part of updates
            updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

            return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """Approximation to the reconstruction error

        Note that this function requires the pre-sigmoid activation as
        input.  To understand why this is so you need to understand a
        bit about how Theano works. Whenever you compile a Theano
        function, the computational graph that you pass as input gets
        optimized for speed and stability.  This is done by changing
        several parts of the subgraphs with others.  One such
        optimization expresses terms of the form log(sigmoid(x)) in
        terms of softplus.  We need this optimization for the
        cross-entropy since sigmoid of numbers larger than 30. (or
        even less then that) turn to 1. and numbers smaller than
        -30. turn to 0 which in terms will force theano to compute
        log(0) and therefore we will get either -inf or NaN as
        cost. If the value is expressed in terms of softplus we do not
        get this undesirable behaviour. This optimization usually
        works fine, but here we have a special case. The sigmoid is
        applied inside the scan op, while the log is
        outside. Therefore Theano will only see log(scan(..)) instead
        of log(sigmoid(..)) and will not apply the wanted
        optimization. We can not go and replace the sigmoid in scan
        with something else also, because this only needs to be done
        on the last step. Therefore the easiest and more efficient way
        is to get also the pre-sigmoid activation as an output of
        scan, and apply both the log and sigmoid outside scan such
        that Theano can catch and optimize the expression.
        """
        cross_entropy = T.mean(
            T.sum(
                self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )
        return cross_entropy


def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')


def test_rbm(learning_rate=0.1, training_epochs=15, dataset='mnist.pkl.gz', batch_size=20, n_chains=20,
             n_samples=10, output_folder='rbm_plots', n_hidden=500):
    """
        Demonstrate how to train and afterwards sample from it using Theano.

        This is demonstrated on MNIST.

        :param learning_rate: learning rate used for training the RBM

        :param training_epochs: number of epochs used for training

        :param dataset: path the the pickled dataset

        :param batch_size: size of a batch used to train the RBM

        :param n_chains: number of parallel Gibbs chains to be used for sampling

        :param n_samples: number of samples to plot for each chain

    """
    with gzip.open('/home/aurora/hdd/workspace/PycharmProjects/data/MNIST/mnist.pkl.gz', 'rb') as f:
        train_set, validate_set, test_set = cPickle.load(f)

    train_set_x, train_set_y = shared_dataset(train_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
    # allocate symbolic variables for the data
    index = T.lscalar()     # index to a [mini]batch
    x = T.matrix('x')       # the data is presented as rasterized images

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2**30))

    # initialize storage for the persistent chain (state = hidden layer of chain)
    persistent_chain = theano.shared(np.zeros((batch_size, n_hidden), dtype=theano.config.floatX), borrow=True)
    # construct the RBM class
    rbm = RBM(input=x, n_visible=28*28, n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)
    # get the cost and the gradient corresponding to one step of CD-15
    # cost, updates = rbm.get_cost_updates(lr=learning_rate, persistent=persistent_chain, k=15)
    cost, updates = rbm.get_cost_updates(lr=learning_rate, k=1)

    #################################
    #     Training the RBM          #
    #################################
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    os.chdir(output_folder)

    # start-snippet-5
    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function([index], cost, updates=updates,
                                givens={x: train_set_x[index*batch_size:(index+1)*batch_size]}, name='train_rbm')

    plotting_time = 0.
    start_time = timeit.default_timer()

    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through the training set
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_rbm(batch_index)]
            # print mean_cost
            # print rbm.w.get_value()

        print 'Training epoch %d, cost is ' % epoch, np.mean(mean_cost)

        # Plot filters after each training epoch
        plotting_start = timeit.default_timer()
        # construct image from the weight matrix
        image = Image.fromarray(
            tile_raster_image(x=rbm.w.get_value(borrow=True).T, img_shape=(28, 28), tile_shape=(10, 10),
                              tile_spacing=(1, 1))
        )
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop-plotting_start)
    end_time = timeit.default_timer()

    pretraining_time = (end_time-start_time) - plotting_time
    print ('Training took %f minutes' % (pretraining_time / 60.))

    # saving the weights and bias
    f = file('/home/aurora/hdd/workspace/PycharmProjects/data/rbm_weights_cd.save', 'wb')
    for obj in [rbm.w, rbm.hbias, rbm.vbias]:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    # end-snippet-5 start-snippet-6
    #################################
    #     Sampling from the RBM     #
    #################################
    # find out the number of test samples
    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]

    # pick random test examples, with which to initialize the persistent chain
    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(
        np.asarray(
            test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
            dtype=theano.config.floatX
        )
    )
    # end-snippet-6 start-snippet-7
    plot_every = 1000
    # define one step of Gibbs sampling (mf = mean-field) define a
    # function that does `plot_every` steps before returning the
    # sample for plotting
    (
        [
            presig_hids,
            hid_mfs,
            hid_samples,
            presig_vis,
            vis_mfs,
            vis_samples
        ],
        updates
    ) = theano.scan(
        rbm.gibbs_vhv,
        outputs_info=[None, None, None, None, None, persistent_vis_chain],
        n_steps=plot_every
    )

    # add to updates the shared variable that takes care of our persistent
    # chain :.
    updates.update({persistent_vis_chain: vis_samples[-1]})
    # construct the function that implements our persistent chain.
    # we generate the "mean field" activations for plotting and the actual
    # samples for reinitializing the state of our persistent chain
    sample_fn = theano.function(
        [],
        [
            vis_mfs[-1],
            vis_samples[-1]
        ],
        updates=updates,
        name='sample_fn'
    )

    # create a space to store the image for plotting ( we need to leave
    # room for the tile_spacing as well)
    image_data = np.zeros(
        (29 * n_samples + 1, 29 * n_chains - 1),
        dtype='uint8'
    )
    for idx in xrange(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        vis_mf, vis_sample = sample_fn()
        print ' ... plotting sample ', idx
        print 'the id is generate each time??? ', test_idx
        image_data[29 * idx:29 * idx + 28, :] = tile_raster_image(
            x=vis_mf,
            img_shape=(28, 28),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1)
        )

    # construct image
    image = Image.fromarray(image_data)
    image.save('samples.png')
    # end-snippet-7
    os.chdir('../')


if __name__=='__main__':
    test_rbm()