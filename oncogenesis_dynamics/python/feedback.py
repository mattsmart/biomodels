# feedback.py
# shapes of feedback assigned in params.py used for the dynamics

import matplotlib.pyplot as plt
import numpy as np

from constants import ODE_SYSTEMS, PARAMS_ID, PARAMS_ID_INV, HILL_Z0_RATIO, HILL_Y0_PLUS_Z0_RATIO, HILL_EXP, \
                      PARAMS_ID_INV, PARAM_GAMMA, DEFAULT_FEEDBACK_SHAPE, FEEDBACK_SHAPES


def hill_increase(param_to_modify, coord, N, hill_exp=HILL_EXP, hill_ratio=HILL_Z0_RATIO):
    """
    param_to_modify: e.g. alpha_plus_0 (value without feedback)
    coord: state coordinate e.g. z=50
    N: param N e.g. N=10,000
    """
    feedbackval = param_to_modify * (1 + coord ** hill_exp / (coord ** hill_exp + (hill_ratio * N) ** hill_exp))
    return feedbackval


def hill_decrease(param_to_modify, coord, N, hill_exp=HILL_EXP, hill_ratio=HILL_Z0_RATIO):
    """
    param_to_modify: e.g. alpha_plus_0 (value without feedback)
    coord: state coordinate e.g. z=50
    N: param N e.g. N=10,000
    """
    feedbackval = param_to_modify * (hill_ratio * N) ** hill_exp / (coord ** hill_exp + (hill_ratio * N) ** hill_exp)
    return feedbackval


def plot_all_feedbacks():
    # value settings
    param_to_modify = 10.0
    N = 100
    coord_range = np.linspace(0, N, 100)
    # plot settings
    x_label = 'state coordinate (e.g. z)'
    y_label = 'feedback value of parameter'
    feedback_labels = ['hill_increase', 'hill_decrease']
    feedback_fns = [hill_increase, hill_decrease]
    # subplot settings
    numrow = 2
    numcol = 2
    #plt.subplots(numrow, numcol, sharex='col', sharey='row')
    fig, axarr = plt.subplots(numrow, numcol)
    # plotting
    for i in xrange(numrow):
        for j in xrange(numcol):
            idx = numrow*i + j
            if idx >= len(feedback_labels):
                break
            feedback_fn = feedback_fns[idx]

            param_range = [feedback_fn(coord, param_to_modify, N) for coord in coord_range]
            axarr[i,j].plot(coord_range, param_range)
            axarr[i,j].set_title('Feedback: %s' % feedback_labels[idx])
            axarr[i,j].set_xlabel(x_label)
            axarr[i,j].set_ylabel(y_label)
    plt.show()
    return

if __name__ == '__main__':
    plot_all_feedbacks()
