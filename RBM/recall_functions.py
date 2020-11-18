import matplotlib.pyplot as plt
import numpy as np


def hamming(s1, s2):
    """Calculate the Hamming distance between two bit lists"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def update_site_glauber(state, site, intxn_matrix, urand, beta):
    total_field = np.dot(intxn_matrix, state)
    prob_on_after_timestep = 1 / (
            1 + np.exp(-2 * beta * total_field))  # probability that site i will be "up" after the timestep
    if prob_on_after_timestep > urand:
        state[site] = 1.0
    else:
        state[site] = -1.0
    return state


def update_state_noise(state, intxn_matrix, beta, async_batch=True):
    sites = range(N)
    rsamples = np.random.rand(
        N)  # optimized: pass one to each of the N single spin update calls  TODO: benchmark vs intels
    if async_batch:
        shuffle(sites)  # randomize site ordering each timestep updates
    else:
        sites = [int(N * u) for u in
                 np.random.rand(N)]  # this should be 5-10% percent faster, pick N sites at random with possible repeats

    state_end = np.copy(state)
    for idx, site in enumerate(sites):  # TODO: parallelize approximation
        state_end = update_site_glauber(state_end, site, intxn_matrix, rsamples[idx], beta)

    return state_end


def update_state_deterministic(state, intxn_matrix):
    total_field = np.dot(intxn_matrix, state)
    state = np.sign(total_field)
    # TODO care for zeros
    if any(state == 0):
        assert 1 == 2
    return state


def plot_confusion_matrix_recall(confusion_matrix, classlabels=list(range(10)), title='', save=None):
    # Ref: https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
    import seaborn as sn
    import pandas as pd

    ylabels = classlabels
    if confusion_matrix.shape[1] == len(ylabels) + 1:
        xlabels = ylabels + ['Other']
    else:
        xlabels = ylabels
    df_cm = pd.DataFrame(confusion_matrix, index=ylabels, columns=xlabels)

    plt.figure(figsize=(11, 7))
    sn.set(font_scale=1.2)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap='Blues', fmt='d')  # font size
    plt.gca().set(xlabel=r'$Fixed point$', ylabel=r'$True label$')
    plt.title(title)
    if save is None:
        plt.show()
    else:
        plt.savefig(save)
    return


def build_J_from_xi(xi, remove_diag=False):
    A = np.dot(xi.T, xi)
    A_inv = np.linalg.inv(A)
    intxn_matrix = np.dot(xi,
                          np.dot( A_inv, xi.T))
    if remove_diag:
        for i in range(intxn_matrix.shape[0]):
            intxn_matrix[i, i] = 0.0
    return intxn_matrix
