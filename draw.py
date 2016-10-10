import numpy as np
import theano
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

import nice
import data


def plot_digits(digit_array):
    """Visualizes each example in digit_array.

    Note: N is the number of examples
          and M is the number of features per example.

    Inputs:
        digits: N x M array of pixel intensities.
    """

    CLASS_EXAMPLES_PER_PANE = 4

    # assume two evenly split classes
    examples_per_class = int(digit_array.shape[0]/2)
    num_panes = int(np.ceil(float(examples_per_class)/CLASS_EXAMPLES_PER_PANE))

    for pane in range(num_panes):
        print("Displaying pane {}/{}".format(pane+1, num_panes))

        top_start = pane*CLASS_EXAMPLES_PER_PANE
        top_end = min((pane+1)*CLASS_EXAMPLES_PER_PANE, examples_per_class)
        top_pane_digits = extract_digits(digit_array, top_start, top_end)

        bottom_start = top_start + examples_per_class
        bottom_end = top_end + examples_per_class
        bottom_pane_digits = extract_digits(digit_array, bottom_start, bottom_end)

        show_pane(top_pane_digits, bottom_pane_digits)


def extract_digits(digit_array, start_index, end_index):
    """Returns a list of 28 x 28 pixel intensity arrays starting
    at start_index and ending at end_index.
    """

    digits = []
    for index in range(start_index, end_index):
        digits.append(extract_digit_pixels(digit_array, index))

    return digits


def extract_digit_pixels(digit_array, index):
    """Extracts the 28 x 28 pixel intensity array at the specified index.
    """

    return digit_array[index].reshape(28, 28)


def show_pane(top_digits, bottom_digits):
    """Displays two rows of digits on the screen.
    """

    all_digits = top_digits + bottom_digits
    fig, axes = plt.subplots(nrows = 2, ncols = int(len(all_digits)/2))
    for axis, digit in zip(axes.reshape(-1), all_digits):
        axis.imshow(digit, interpolation='nearest', cmap=plt.gray())
        axis.axis('off')
    plt.show()

if __name__ == '__main__':
    params, x_1, x_2, y, pred_input_1, pred_input_2, f_pred, f_log_prob, cost = nice.build_model()
    f_pred_input_1 = theano.function([y], pred_input_1, name="pred_input_1")
    f_pred_input_2 = theano.function([y], pred_input_2, name="pred_input_2")

    pp = np.load('weights.npz')

    for k in pp:
        params[k].set_value(pp[k])

    batch_size = 40
    output_size = 2*392
    y = np.array(np.random.logistic(size=[batch_size, output_size]), dtype=theano.config.floatX)
    input_1 = f_pred_input_1(y)
    input_2 = f_pred_input_2(y)
    plot_digits(data.recombine_data(input_1, input_2))
