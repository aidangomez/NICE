import theano
import numpy

import nice
import data
import draw

from theano import config
import time


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)

def adadelta(lr, tparams, grads, x_1, x_2, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x_1, x_2], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-theano.tensor.sqrt(ru2 + 1e-6) / theano.tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update

def adam(loss, all_params, x_1, x_2, learning_rate=1e-3, b1=0.9, b2=0.01, e=1e-4, gamma=1-1e-8):
    updates = []
    all_grads = theano.grad(loss, all_params)
    alpha = learning_rate
    t = theano.shared(numpy.float32(1))
    b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)

    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(numpy.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))
        v_previous = theano.shared(numpy.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))

        m = b1_t*m_previous + (1 - b1_t)*g                             # (Update biased first moment estimate)
        v = b2*v_previous + (1 - b2)*g**2                              # (Update biased second raw moment estimate)
        m_hat = m / (1-b1**t)                                          # (Compute bias-corrected first moment estimate)
        v_hat = v / (1-b2**t)                                          # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - (alpha * m_hat) / (theano.tensor.sqrt(v_hat) + e) #(Update parameters)

        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta) )
    updates.append((t, t + 1.))

    f_update = theano.function([x_1, x_2], [], updates=updates,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_update


def pred_error(f_pred_prob, x_1, x_2, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        pred = f_pred_prob(x_1[valid_index], x_2[valid_index])
        valid_err += pred

    valid_err = numpy_floatX(valid_err) / len(x_1)

    return valid_err

def train_nice(
    max_epochs=3,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    validFreq=333,  # Compute the validation error after this number of update.
    batch_size=40,  # The batch size during training.
    lrate=0.001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    valid_batch_size=64,  # The batch size used for validation/test set.

    # Parameter for extra option
    noise_std=0.,
    test_size=-1,  # If >0, we keep only this number of test example.
):

    # CPU-side Data
    (train_cpu, _), (valid_cpu, _), (test_cpu, _) = data.get_dataset()

    # Partitioned GPU-side Data
    train_gpu_1, train_gpu_2 = data.partition_data(train_cpu)
    valid_gpu_1, valid_gpu_2 = data.partition_data(valid_cpu)
    test_gpu_1, test_gpu_2 = data.partition_data(test_cpu)

    print(train_gpu_1.shape)

    print('Building model')
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params, x_1, x_2, y, pred_input_1, pred_input_2, f_pred, f_log_prob, cost = nice.build_model()

    f_cost = theano.function([x_1, x_2], cost, name='f_cost')
    f_pred_input_1 = theano.function([y], pred_input_1, name="pred_input_1")
    f_pred_input_2 = theano.function([y], pred_input_2, name="pred_input_2")

    grads = theano.tensor.grad(cost, wrt=list(params.values()))
    f_grad = theano.function([x_1, x_2], grads, name='f_grad')

    lr = theano.tensor.scalar(name='lr')
    f_update = adam(cost, list(params.values()), x_1, x_2)#adadelta(lr, params, grads, x_1, x_2, cost)
    f_grad_shared = f_cost

    print('Optimization')

    kf_valid = data.get_minibatches_idx(len(valid_gpu_1), valid_batch_size)
    kf_test = data.get_minibatches_idx(len(test_gpu_1), valid_batch_size)

    print("%d train examples" % len(train_gpu_1))
    print("%d valid examples" % len(valid_gpu_1))
    print("%d test examples" % len(test_gpu_1))

    history_errs = []
    best_p = None
    bad_count = 0

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = data.get_minibatches_idx(len(train_gpu_1), batch_size, shuffle=True)

            for (_, train_index) in kf:
                uidx += 1

                # Select the random examples for this minibatch
                x_1 = numpy.array([train_gpu_1[t]for t in train_index])
                x_2 = numpy.array([train_gpu_2[t]for t in train_index])


                n_samples += x_1.shape[0]

                cost = f_grad_shared(x_1, x_2)
                log_prob = f_log_prob(x_1, x_2)
                f_update(x_1, x_2)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'Log Prob ', log_prob)

                if numpy.mod(uidx, validFreq) == 0:
                    train_err = pred_error(f_log_prob, train_gpu_1, train_gpu_2, kf)
                    valid_err = pred_error(f_log_prob, valid_gpu_1, valid_gpu_2,
                                           kf_valid)
                    test_err = pred_error(f_log_prob, test_gpu_1, test_gpu_2, kf_test)

                    print( ('Train ', train_err, 'Valid ', valid_err,
                           'Test ', test_err) )

            y = numpy.random.logistic(size=[x_1.shape[0], x_1.shape[1]+x_2.shape[1]])
            input_1 = f_pred_input_1(y)
            input_2 = f_pred_input_2(y)
            draw.plot_digits(data.recombine_data(input_1, input_2), "epoch_%d.png" % eidx)

            print('Seen %d samples' % n_samples)

            if estop:
                break

    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()

    kf_train_sorted = data.get_minibatches_idx(len(train_gpu_1), batch_size)
    train_err = pred_error(f_log_prob, train_gpu_1, train_gpu_2, kf_train_sorted)
    valid_err = pred_error(f_log_prob, valid_gpu_1, valid_gpu_2, kf_valid)
    test_err = pred_error(f_log_prob, test_gpu_1, test_gpu_2, kf_test)

    print( 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err )
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))


    return train_err, valid_err, test_err

if __name__ == '__main__':
    train_nice()
