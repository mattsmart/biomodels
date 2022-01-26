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


def detect_oscillations_scipy(times, traj, min_height=None, max_valley=None, show=False):
    """
    Uses scipy "find_peaks" https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    """
    assert times.shape == traj.shape

    peaks, peaks_properties = signal.find_peaks(
        traj, height=min_height, threshold=None, distance=None, prominence=None, wlen=None, plateau_size=None)
    valleys, valleys_properties = signal.find_peaks(
        -1 * traj, height=max_valley, threshold=None, distance=None, prominence=None, wlen=None, plateau_size=None)

    # based on the peaks and oscillations, report the event times
    t_events = [times[peaks[i]] for i in range(1, len(peaks))]
    duration_cycles = [times[peaks[i]] - times[peaks[i-1]] for i in range(1, len(peaks))]
    num_oscillations = len(t_events)

    if show:
        plt.plot(times, traj, '-', c='k')
        plt.plot(times[peaks], traj[peaks], 'o', c='red')
        plt.plot(times[valleys], traj[valleys], 'o', c='blue')
        for idx in range(num_oscillations):
            plt.axvline(t_events[idx], linestyle='--', c='gray')
        plt.show()

    return num_oscillations, t_events, duration_cycles


if __name__ == '__main__':
    times = np.loadtxt('input' + os.sep + 'traj_times.txt')
    r = np.loadtxt('input' + os.sep + 'traj_x.txt')

    r_choice = r[:, 1]  # try idx 0 or 1 (active/total cyclin)

    peaks, properties = signal.find_peaks(
        r_choice, height=None, threshold=None, distance=None, prominence=None, width=None, wlen=None,
        rel_height=0.5, plateau_size=None)
    valleys, properties = signal.find_peaks(
        -1 * r_choice, height=None, threshold=None, distance=None, prominence=None, width=None, wlen=None,
        rel_height=0.5, plateau_size=None)
    print(peaks)
    print(properties)

    num_oscillations, t_events, duration_cycles = detect_oscillations_scipy(times, r_choice, show=True)

    print('Timeseries has %d oscillations' % num_oscillations)
    print('Oscillation info:')
    for idx in range(num_oscillations):
        print('\t(%d of %d) - Time of event:', (idx, num_oscillations, t_events[idx]))
        print('\t(%d of %d) - Period of cycle:', (idx, num_oscillations, duration_cycles[idx]))
