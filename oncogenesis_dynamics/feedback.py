# feedback.py
# shapes of feedback assigned in params.py used for the dynamics
# TODO implement tanh (beta is param? slide center param?)
# TODO implement relu/pwLinear (fewer params than slope, start, end)

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


def tanh_unit(coord_normed, rate=5.0, switchpoint=SWITCHING_RATIO):
    """
    Eegularized unit between 0 and 1 effectively 0.5*(1+tan(x))=1/(1+exp(-2x))
    As coord -> infty, val -> 1
    As coord -> -infty, val -> 0
    Switchpoint defines the midpoint of the s-shape
    Rate define the steepness: rate -> infty should resemble step function at coord = midpoint
    Defaults: switchpoint ~ 0.5, rate ~ 5.0 so that as coord -> 0, val -> 0
    Notes:
        - could additionally perform val ** (1/hill_exp) to modify the switching behaviour
    """
    x = rate * (coord_normed - switchpoint)
    return 1 / (1 + np.exp(-2*x))                     # note this is 0.5*(1+tan(x))


def tanh_increase(param_to_modify, coord, N, rate=5.0, switchpoint=SWITCHING_RATIO, multiplier=MULT_INC):
    """
    param_to_modify: e.g. alpha_plus_0 (value without feedback)
    coord: state coordinate e.g. z=50
    N: param N e.g. N=10,000
    multiplier: sets scale of increase, e.g. multiplier=k corresponds to saturation at (k+1)*param_to_modify
    """
    f = coord / float(N)
    tanhcurve = tanh_unit(f, rate=rate, switchpoint=switchpoint)
    mult_factor = (multiplier - 1)
    feedbackval = param_to_modify * (1 + mult_factor * tanhcurve)
    return feedbackval


def tanh_decrease(param_to_modify, coord, N, rate=5.0, switchpoint=SWITCHING_RATIO, multiplier=MULT_DEC):
    """
    param_to_modify: e.g. alpha_plus_0 (value without feedback)
    coord: state coordinate e.g. z=50
    N: param N e.g. N=10,000
    multiplier: param_to_modify gets "multiplier" times smaller as coord -> N
    """
    f = coord / float(N)
    tanhcurve = tanh_unit(f, rate=rate, switchpoint=switchpoint)
    mult_factor = 1/multiplier - 1
    feedbackval = param_to_modify * (1 + mult_factor * tanhcurve)
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
                       'step_increase', 'step_decrease',
                       'tanh_increase', 'tanh_decrease']
    feedback_fns = [hill_orig_increase, hill_orig_decrease,
                    hill_increase, hill_decrease,
                    step_increase, step_decrease,
                    tanh_increase, tanh_decrease]
    # subplot settings
    numrow = 4
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
            print feedback_labels[idx], param_range[0], param_range[-1]
    plt.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    plot_all_feedbacks()
