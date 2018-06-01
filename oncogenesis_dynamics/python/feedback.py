# feedback.py
# shapes of feedback assigned in params.py used for the dynamics

import matplotlib.pyplot as plt
import numpy as np

from constants import HILLORIG_Z0_RATIO, HILL_EXP, MULT_INC, MULT_DEC, SWITCHING_RATIO


def hill_orig_increase(param_to_modify, coord, N, hill_exp=HILL_EXP, hill_ratio=HILLORIG_Z0_RATIO):
    """
    param_to_modify: e.g. alpha_plus_0 (value without feedback)
    coord: state coordinate e.g. z=50
    N: param N e.g. N=10,000
    """
    feedbackval = param_to_modify * (1 + coord ** hill_exp / (coord ** hill_exp + (hill_ratio * N) ** hill_exp))
    return feedbackval


def hill_orig_decrease(param_to_modify, coord, N, hill_exp=HILL_EXP, hill_ratio=HILLORIG_Z0_RATIO):
    """
    param_to_modify: e.g. alpha_plus_0 (value without feedback)
    coord: state coordinate e.g. z=50
    N: param N e.g. N=10,000
    """
    feedbackval = param_to_modify * (hill_ratio * N) ** hill_exp / (coord ** hill_exp + (hill_ratio * N) ** hill_exp)
    return feedbackval


def hill_increase(param_to_modify, coord, N, hill_exp=HILL_EXP, hill_ratio=SWITCHING_RATIO, multiplier=MULT_INC):
    """
    param_to_modify: e.g. alpha_plus_0 (value without feedback)
    coord: state coordinate e.g. z=50
    N: param N e.g. N=10,000
    multiplier: sets scale of increase, e.g. multiplier=k corresponds to saturation at (k+1)*param_to_modify
    """
    f = coord / float(N)
    ratio = f ** hill_exp /(hill_ratio ** hill_exp + f ** hill_exp)
    mult_factor = (multiplier - 1) * (hill_ratio ** hill_exp + 1)
    feedbackval = param_to_modify * (1 + mult_factor * ratio)
    return feedbackval


def hill_decrease(param_to_modify, coord, N, hill_exp=HILL_EXP, hill_ratio=SWITCHING_RATIO, multiplier=MULT_DEC):
    """
    param_to_modify: e.g. alpha_plus_0 (value without feedback)
    coord: state coordinate e.g. z=50
    N: param N e.g. N=10,000
    multiplier: param_to_modify gets "multiplier" times smaller as coord -> N
    """
    f = coord / float(N)
    eps = 1 / (1 + (1/hill_ratio) ** hill_exp)
    # first term matches original hill feedback
    term1 = 1 / (1 + (f/hill_ratio) ** hill_exp)
    # second term added to make it saturate not near 0 (i.e. at 1/mult)
    ratio = f ** hill_exp /(hill_ratio ** hill_exp + f ** hill_exp)
    mult_factor = (hill_ratio ** hill_exp + 1) * (1/multiplier - eps)
    term2 = ratio*mult_factor

    feedbackval = param_to_modify*(term1 + term2)  # term1 goes from 1 to eps, term2 goes from eps to 1/gamma
    return feedbackval


def step_increase(param_to_modify, coord, N, step_ratio=SWITCHING_RATIO, multiplier=MULT_INC):
    """
    param_to_modify: e.g. alpha_plus_0 (value without feedback)
    coord: state coordinate e.g. z=50
    N: param N e.g. N=10,000
    """
    step_coord = N * step_ratio
    if coord > step_coord:
        feedbackval = param_to_modify * multiplier
    elif coord == step_coord:
        feedbackval = param_to_modify * (multiplier + 1) / 2  # note this appears to be unnecessary
    else:
        feedbackval = param_to_modify
    return feedbackval


def step_decrease(param_to_modify, coord, N, step_ratio=SWITCHING_RATIO, multiplier=MULT_DEC):
    """
    param_to_modify: e.g. alpha_plus_0 (value without feedback)
    coord: state coordinate e.g. z=50
    N: param N e.g. N=10,000
    """
    step_coord = N * step_ratio
    if coord > step_coord:
        feedbackval = param_to_modify / multiplier
    elif coord == step_coord:
        feedbackval = param_to_modify / ((multiplier + 1) / 2)  # note this appears to be unnecessary
    else:
        feedbackval = param_to_modify
    return feedbackval


def plot_all_feedbacks():
    # value settings
    param_to_modify = 10.0
    N = 100
    coord_range = np.linspace(0, N, 100)
    # plot settings
    x_label = 'state coordinate (e.g. z)'
    y_label = 'value (e.g. alpha(z))'
    feedback_labels = ['hill_orig_increase', 'hill_orig_decrease',
                       'hill_increase', 'hill_decrease',
                       'step_increase', 'step_decrease']
    feedback_fns = [hill_orig_increase, hill_orig_decrease,
                    hill_increase, hill_decrease,
                    step_increase, step_decrease]
    # subplot settings
    numrow = 3
    numcol = 2
    #plt.subplots(numrow, numcol, sharex='col', sharey='row')
    fig, axarr = plt.subplots(numrow, numcol)
    # plotting
    for i in xrange(numrow):
        for j in xrange(numcol):
            idx = numcol*i + j
            feedback_fn = feedback_fns[idx]
            param_range = [feedback_fn(param_to_modify, coord, N) for coord in coord_range]
            axarr[i,j].plot(coord_range, param_range)
            axarr[i,j].set_title('Feedback: %s' % feedback_labels[idx])
            axarr[i,j].set_xlabel(x_label)
            axarr[i,j].set_ylabel(y_label)
    plt.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    plot_all_feedbacks()