import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.signal as signal


def detect_oscillations_manual(times, traj, expect_lower, expect_upper, buffer=10):
    """
    buffer is plus/minus "buffer" percent from the expected upper/lower points of the oscillation turning points

    Option 1) detect sudden changes in traj (i.e. sharp peaks/valleys --> oscillation turning points)
    Option 2) detect "crossings" past some reference point (e.g. the midpoint of expect_lower, expect_upper)

    See https://docs.scipy.org/doc/scipy/reference/signal.html for numerous options
    """
    # TODO - for now will use scipy
    assert times.shape == traj.shape
    assert expect_lower < expect_upper
    num_oscillations = 0
    # Step 1) detect all sudden changes with dxdt > threshold
    #traj_shift =

    return num_oscillations


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
    events_idx = [peaks[i] - buffer for i in range(1, len(peaks))]
    events_times = [times[events_idx[i]] for i in range(len(events_idx))]
    duration_cycles = [times[events_idx[i]] - times[events_idx[i-1]] for i in range(len(events_idx))]
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
    times = np.loadtxt('input' + os.sep + 'traj_times.txt')
    r = np.loadtxt('input' + os.sep + 'traj_x.txt')
    r_choice = r[:, 1]  # try idx 0 or 1 (active/total cyclin)

    num_oscillations, events_idx, events_times, duration_cycles = detect_oscillations_scipy(times, r_choice, show=True)

    print('\nTimeseries has %d oscillations' % num_oscillations)
    print('Oscillation info:')
    for idx in range(num_oscillations):
        print('\t(%d of %d) - Index of event: %d:' % (idx, num_oscillations, events_idx[idx]))
        print('\t(%d of %d) - Time of event: %.2f:' % (idx, num_oscillations, events_times[idx]))
        print('\t(%d of %d) - Period of cycle: %.2f' % (idx, num_oscillations, duration_cycles[idx]))

    # Now backtest by iteratively truncating the function as might occur during oscillator network trajectory
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

    # TODO why is first period negative ?
