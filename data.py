import pickle
import gzip
import numpy
from collections import OrderedDict
import theano


def get_dataset():
    f = gzip.open('mnist.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train_set, valid_set, test_set = u.load()
    f.close()

    return train_set, valid_set, test_set

def shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue
    return shared_x, theano.tensor.cast(shared_y, 'int32')

def partition_data(data):
    return data[:, 0::2], data[:, 1::2]

def recombine_data(data_1, data_2):
    output = theano.tensor.zeros([data_1.shape[0], data_1.shape[1] + data_2.shape[1]], dtype=theano.config.floatX)
    output = theano.tensor.set_subtensor(output[:, 0::2], data_1)
    output = theano.tensor.set_subtensor(output[:, 1::2], data_2)

    return output

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def unzip_params(params):
    new_params = OrderedDict()
    for kk, vv in params.items():
        new_params[kk] = vv.get_value()
    return new_params
