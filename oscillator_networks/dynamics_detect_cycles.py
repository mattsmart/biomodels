import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.signal as signal


def detect_oscillations_manual(times, traj, expect_lower, expect_upper, show=False):
    """
    buffer is plus/minus "buffer" percent from the expected upper/lower points of the oscillation turning points

    Option 1) detect sudden changes in traj (i.e. sharp peaks/valleys --> oscillation turning points)
    Option 2) detect "crossings" past some reference point (e.g. the midpoint of expect_lower, expect_upper)

    Current approach:
    - have fixed y value which is our crossing criterion
    - shift traj st. zero crossings correspond to events
    - multiply adjacent items in array and find argwhere negative (these will be zero crossings)

    See https://docs.scipy.org/doc/scipy/reference/signal.html for numerous options

    Returns:
        num_oscillations   - "k" int
        events_idx         - k-list of int
        events_times       - k-list of float
        duration_cycles    - k-list of float
    """
    assert times.shape == traj.shape
    assert expect_lower <= expect_upper
    midpoint = 0.5 * (expect_upper + expect_lower)

    """ Event rules: first event occurs
    Rule 1: after at least two "crossings"
    Rule 2: before the second "peak"
    Rule 3: around the second "self-similar" point
    """

    # Step 0) detect crossings (unfiltered) of "the midpoint"
    # when diff prod elements are zero or negative, there is a crossing event
    traj_shifted_midpoint = traj - midpoint
    traj_diff_prod_midpoint = traj_shifted_midpoint[0:-1] * traj_shifted_midpoint[1:]
    cross_indices_midpoint = np.where(traj_diff_prod_midpoint < 0)[0]

    events_idx = []
    # Geometric constraint - Cycle can only occur if there are 2 or more midpoint crossings
    if len(cross_indices_midpoint) >= 2:
        # Step 1) detect crossings (unfiltered) of "the initial point"
        # when diff prod elements are zero or negative, there is a crossing event
        traj_shifted_init = traj - traj[0]  # TODO plus or minus epsilon?
        traj_diff_prod_init = traj_shifted_init[0:-1] * traj_shifted_init[1:]
        return_indices_init = np.where(traj_diff_prod_init < 0)[0]

    # Geometric constraint - Cycle can only occur if there are 2 or more "returns" to initial point
    if len(return_indices_init) >= 2:
        # Step 1) detect events -- requires two subsequent signed crossings (unfiltered)
        # TODO note -- currently ignoring sign of crossings for efficiency; could compare traj_diff_prod[cross_idx] though
        # Implementation: first period occurs AFTER the first peak, but BEFORE the second peak.
        # Likewise for subsequent periods.

        # Note A: consider sin(2 pi t); the first "event" should be called at t = 1, but we will have either two OR
        #  three zero crossings by the time t=1 (depending on exact starting point). However, subsequent cycles will
        #  have two zero crossings in between.

        # Step 3) filtering out spurious events
        # TODO make use of upper/lower in filtering or remove them as args and just pass center line?
        #  Also, have filter for minimum event timing/period? Extra arg for detect call?
        # TODO combine with scipy so 2 extrema (1 peak/valley) is a requirement for a cycle?
        # ...

        # Step 4) shifting the "end of period" point to some standardized spot
        # ...
        events_idx = return_indices_init[1::2]

    # Step 5) output packaging
    events_times = [times[events_idx[i]] for i in range(len(events_idx))]
    duration_cycles = [events_times[i] - events_times[i-1] for i in range(1, len(events_idx))]
    num_oscillations = len(events_idx)

    if show:
        print("in show...", times.shape, times[0:3], times[-3:])
        plt.plot(times, traj, '-', c='k')
        plt.plot(times[events_idx], traj[events_idx], 'o', c='red')
        for idx in range(num_oscillations):
            plt.axvline(events_times[idx], linestyle='--', c='gray')
        plt.show()

    return num_oscillations, events_idx, events_times, duration_cycles


def detect_oscillations_scipy(times, traj, min_height=None, max_valley=None, show=False, buffer=1):
    """
    Uses scipy "find_peaks" https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    Returns:
        num_oscillations   - "k" int
        events_idx         - k-list of int
        events_times       - k-list of float
        duration_cycles    - k-list of float
    """
    assert times.shape == traj.shape

    peaks, peaks_properties = signal.find_peaks(
        traj, height=min_height, threshold=None, distance=None, prominence=None, wlen=None, plateau_size=None)
    valleys, valleys_properties = signal.find_peaks(
        -1 * traj, height=max_valley, threshold=None, distance=None, prominence=None, wlen=None, plateau_size=None)

    # based on the peaks and oscillations, report the event times
    duration_cycles = [times[peaks[i]] - times[peaks[i-1]] for i in range(1, len(peaks))]
    events_idx = [peaks[i] - buffer for i in range(1, len(peaks))]
    events_times = [times[events_idx[i]] for i in range(len(events_idx))]
    num_oscillations = len(events_idx)

    if show:
        print("in show...", times.shape, times[0:3], times[-3:])
        plt.plot(times, traj, '-', c='k')
        plt.plot(times[peaks], traj[peaks], 'o', c='red')
        plt.plot(times[valleys], traj[valleys], 'o', c='blue')
        for idx in range(num_oscillations):
            plt.axvline(events_times[idx], linestyle='--', c='gray')
        plt.show()

    return num_oscillations, events_idx, events_times, duration_cycles


if __name__ == '__main__':

    # 1) load or generate traj data
    flag_load_test_traj = False
    if flag_load_test_traj:
        times = np.loadtxt('input' + os.sep + 'traj_times.txt')
        r = np.loadtxt('input' + os.sep + 'traj_x.txt')
        r_choice = r[:, 1]  # try idx 0 or 1 (active/total cyclin)
    else:
        times = np.linspace(0, 5.2, 1000)
        r_choice = np.sin(2 * np.pi * times)

    # 2) main detection call
    #num_oscillations, events_idx, events_times, duration_cycles = detect_oscillations_scipy(times, r_choice, show=True)
    num_oscillations, events_idx, events_times, duration_cycles = detect_oscillations_manual(times, r_choice, -1, 1, show=True)

    # 3) prints
    print('\nTimeseries has %d oscillations' % num_oscillations)
    print('Oscillation info:')
    for idx in range(num_oscillations):
        print('\t(%d of %d) - Index of event: %d:' % (idx, num_oscillations, events_idx[idx]))
        print('\t(%d of %d) - Time of event: %.2f:' % (idx, num_oscillations, events_times[idx]))
        print('\t(%d of %d) - Period of cycle: %.2f' % (idx, num_oscillations, duration_cycles[idx]))

    # 3) Backtest - iteratively truncating the function as might occur during oscillator network trajectory
    while num_oscillations > 0:
        print('while.......')
        idx_restart = events_idx[0]
        r_choice = r_choice[idx_restart:]
        times = times[idx_restart:]
        num_oscillations, events_idx, events_times, duration_cycles = detect_oscillations_scipy(times, r_choice, show=True)

        print('\nTimeseries has %d oscillations' % num_oscillations)
        print('Oscillation info:')
        for idx in range(num_oscillations):
            print('\t(%d of %d) - Index of event: %d:' % (idx, num_oscillations, events_idx[idx]))
            print('\t(%d of %d) - Time of event: %.2f:' % (idx, num_oscillations, events_times[idx]))
            print('\t(%d of %d) - Period of cycle: %.2f' % (idx, num_oscillations, duration_cycles[idx]))
