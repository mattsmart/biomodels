import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.signal as signal


def detect_oscillations_manual_v1(times, traj, expect_lower=0, expect_upper=0, show=False):
    show=True
    print(expect_lower, expect_upper)
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

    print("cross_indices_midpoint", cross_indices_midpoint)

    events_idx = []
    duration_cycles = []
    # Geometric constraint - Cycle can only occur if there are 2 or more midpoint crossings
    if len(cross_indices_midpoint) >= 2:
        # Step 1) detect crossings (unfiltered) of "the initial point"
        # when diff prod elements are zero or negative, there is a crossing event
        traj_shifted_init = traj - traj[0]  # TODO plus or minus epsilon?
        traj_diff_prod_init = traj_shifted_init[0:-1] * traj_shifted_init[1:]
        return_indices_init = np.where(traj_diff_prod_init < 0)[0]
        print("return_indices_init", return_indices_init)

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
    num_oscillations = len(events_idx)
    if num_oscillations > 0:
        duration_cycle_first = times[events_idx[0]] - times[0]
        duration_cycles = [duration_cycle_first] + [events_times[i] - events_times[i - 1] for i in range(1, len(events_idx))]

    if show:
        print("in show...", times.shape, times[0:3], times[-3:])
        plt.plot(times, traj, '-', c='k')
        plt.plot(times[events_idx], traj[events_idx], 'o', c='red')
        for idx in range(num_oscillations):
            plt.axvline(events_times[idx], linestyle='--', c='gray')
        plt.axhline(midpoint, linestyle='--', c='gray')
        plt.show()

    return num_oscillations, events_idx, events_times, duration_cycles


def detect_oscillations_manual_2d(times, traj, xlow=0, xhigh=0, ylow=0, yhigh=0, state_xy=[0,1], show=False):
    show=True
    print("detect_oscillations_manual", xlow, xhigh, ylow, yhigh)
    print("todo implement detect_oscillations_manual_2d()")
    assert 1==2
    """
    Inputs:
        time: 1D arr
        traj: nD arr (for now)
        xlow, xhigh: thresholds for x variable [0] cycle detection
        ylow, yhigh: thresholds for y variable [1] cycle detection
    
    TODO split into soft and strict function? call separately from detection styles
    Current approach: designed for style_ode == 'PWL3_swap'
    TODO this strict form is unused (see soft form below)
    - a "standard" cycle requires the following sequence of events
        A: both in any order [xhigh cross from below] and [ylow cross from left]
        THEN
        B: both in any order [xhigh cross from above] and [yhigh cross from left]
        THEN
        C: both in any order [xlow cross from above] and [yhigh cross from right]
        THEN
        D: both in any order [xlow cross from below] and [ylow cross from right]
        THEN 
        A: either of the conditions must be met again
    An event is detected the moment before condition A is satisfied for the second time in the full sequence above
    
    Returns:
        num_oscillations   - "k" int
        events_idx         - k-list of int
        events_times       - k-list of float
        duration_cycles    - k-list of float
    """
    traj_2d = traj[state_xy, :]
    assert times.shape[0] == traj_2d.shape[1]
    assert xlow <= xhigh
    assert ylow <= yhigh
    xmid = 0.5 * (xlow + xhigh)
    ymid = 0.5 * (ylow + yhigh)

    def get_cross_indices(traj_1d, threshold, from_below=True):
        # TODO if this is used in multiple manual fns then move it out (also cleanup/remove the toDOs)
        traj_shifted_threshold = traj_1d - threshold
        traj_diff_prod_threshold = traj_shifted_threshold[0:-1] * traj_shifted_threshold[1:]
        cross_indices_threshold = np.where(traj_diff_prod_threshold <= 0)[0]

        cross_indices_pruned = []
        if from_below:
            for idx in cross_indices_threshold:
                print(idx, traj_shifted_threshold[idx], traj_shifted_threshold[idx + 1])  # toDO remove
                if traj_shifted_threshold[idx] < traj_shifted_threshold[idx + 1]:
                    assert np.sign(traj_shifted_threshold[idx + 1]) == 1  # toDO remove
                    cross_indices_pruned += [idx]
        else:
            for idx in cross_indices_threshold:
                print(idx, traj_shifted_threshold[idx], traj_shifted_threshold[idx + 1])  # toDO remove
                if traj_shifted_threshold[idx] > traj_shifted_threshold[idx + 1]:
                    assert np.sign(traj_shifted_threshold[idx + 1]) == -1  # toDO remove
                    cross_indices_pruned += [idx]

        return cross_indices_pruned

    # TODO revise this 1d code to the stricter 2d case as in the docstring
    A_events = get_cross_indices(traj_2d[0,:], yhigh, from_below=True)
    B_events = get_cross_indices(traj_2d[0,:], yhigh, from_below=False)

    # RULES:
    # - need at least two A events
    # - need A[0] < B[0] < A[1]
    # - return first oscillation info only (for now), with "event_idx = A[1]"

    events_idx = []
    duration_cycles = []
    events_times = []
    if len(A_events) > 1 and len(B_events) > 0:
        A0 = A_events[0]
        A1 = A_events[1]

        # now search for first B0 which occurs after A0
        A0_where_to_put = np.searchsorted(B_events, A0)
        # if A0_where_to_put >= len(B_above), no suitable elements and therefore no valid events
        if A0_where_to_put < len(B_events):
            B_first = B_events[A0_where_to_put]

        if A0 < B_first < A1:
            events_idx = [A1]
            duration_cycles = [times[A1] - times[A0]]
            events_times = [times[A1]]
    num_oscillations = len(events_idx)

    if show:
        # TODO revise plot for 2d case based on phase portrait
        print("in show...", times.shape, times[0:3], times[-3:])
        plt.plot(times, traj, '-', c='k')
        plt.plot(times[events_idx], traj[events_idx], 'o', c='red')
        for idx in range(num_oscillations):
            plt.axvline(events_times[idx], linestyle='--', c='gray')
        plt.axhline(yhigh, linestyle='--', c='gray')
        plt.axhline(ylow, linestyle='--', c='gray')
        plt.show()

    return num_oscillations, events_idx, events_times, duration_cycles


def detect_oscillations_manual(times, traj, xlow=0, xhigh=0, state_choice=None, show=False):
    show = True
    print("detect_oscillations_manual():", "xlow", xlow, "xhigh", xhigh)
    """
    Inputs:
        time: 1D arr
        traj: nD arr
        xlow, xhigh: thresholds for x variable (index controlled by state_choice) cycle detection

    Soft, 1D version of the nD detection sequence is as follows, using only one coordinate (e.g. x or y coord)
    - A: y cross yhigh from below
    - B: y cross ylow from above
    - A (again): y cross yhigh from below  -- event is called the moment before A occurs again
    
    Returns:
        num_oscillations   - "k" int  (currently 0 or 1; this is hard coded to be capped at 1)
        events_idx         - k-list of int
        events_times       - k-list of float
        duration_cycles    - k-list of float
    """
    assert times.shape[0] == traj.shape[1]
    assert xlow <= xhigh
    xmid = 0.5 * (xlow + xhigh)

    def get_cross_indices(traj_1d, threshold, from_below=True):
        traj_shifted_threshold = traj_1d - threshold
        traj_diff_prod_threshold = traj_shifted_threshold[0:-1] * traj_shifted_threshold[1:]
        cross_indices_threshold = np.where(traj_diff_prod_threshold <= 0)[0]

        cross_indices_pruned = []
        if from_below:
            print("From below TRUE")
            print("idx, traj_shifted_threshold[idx], traj_shifted_threshold[idx + 1]")
            for idx in cross_indices_threshold:
                print(idx, traj_shifted_threshold[idx], traj_shifted_threshold[idx + 1])  # toDO remove
                print(idx, traj_1d[idx], traj_1d[idx + 1])  # toDO remove

                if traj_shifted_threshold[idx] < traj_shifted_threshold[idx + 1]:
                    assert np.sign(traj_shifted_threshold[idx + 1]) == 1  # toDO remove
                    cross_indices_pruned += [idx]
        else:
            print("From below FALSE")
            print("idx, traj_shifted_threshold[idx], traj_shifted_threshold[idx + 1]")
            for idx in cross_indices_threshold:
                print(idx, traj_shifted_threshold[idx], traj_shifted_threshold[idx + 1])  # toDO remove
                print(idx, traj_1d[idx], traj_1d[idx + 1])  # toDO remove

                if traj_shifted_threshold[idx] > traj_shifted_threshold[idx + 1]:
                    assert np.sign(traj_shifted_threshold[idx + 1]) == -1  # toDO remove
                    cross_indices_pruned += [idx]

        return cross_indices_pruned

    traj_1d = np.squeeze(traj[state_choice, :])
    print("Collecting A events...")
    A_events = get_cross_indices(traj_1d, xhigh, from_below=True)
    print("Collecting B events...")
    B_events = get_cross_indices(traj_1d, xlow, from_below=False)

    # RULES:
    # - need at least two A events
    # - need A[0] < B[0] < A[1]
    # - return first oscillation info only (for now), with "event_idx = A[1]"
    events_idx = []
    duration_cycles = []
    events_times = []
    if len(A_events) > 1 and len(B_events) > 0:
        A0 = A_events[0]
        A1 = A_events[1]

        # now search for first B0 which occurs after A0
        A0_where_to_put = np.searchsorted(B_events, A0)
        # if A0_where_to_put >= len(B_above), no suitable elements and therefore no valid events
        if A0_where_to_put < len(B_events):
            B_first = B_events[A0_where_to_put]

        if A0 < B_first < A1:
            events_idx = [A1]
            duration_cycles = [times[A1] - times[A0]]
            events_times = [times[A1]]
    num_oscillations = len(events_idx)

    if show:
        print("in show...", times.shape, times[0:3], times[-3:])
        plt.figure(figsize=(5,5))
        plt.plot(times, traj_1d, 'o', linewidth=0.1, c='k')
        plt.plot(times[events_idx], traj_1d[events_idx], 'o', c='red')
        for idx in range(num_oscillations):
            plt.axvline(events_times[idx], linestyle='--', c='gray')
        plt.axhline(xhigh, linestyle='--', c='gray')
        plt.axhline(xlow, linestyle='--', c='gray')
        plt.show()

    return num_oscillations, events_idx, events_times, duration_cycles


def detect_oscillations_scipy(times, traj, state_choice=None, min_height=None, max_valley=None, show=False, buffer=1):
    """
    Uses scipy "find_peaks" https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    Returns:
        num_oscillations   - "k" int
        events_idx         - k-list of int
        events_times       - k-list of float
        duration_cycles    - k-list of float
    """
    assert times.shape[0] == traj.shape[1]
    traj = np.squeeze(traj[state_choice, :])  # work with 1d problem along chosen state variable axis

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
        plt.figure(figsize=(5,5))
        plt.plot(times, traj, '-', c='k')
        plt.plot(times[peaks], traj[peaks], 'o', c='red')
        plt.plot(times[valleys], traj[valleys], 'o', c='blue')
        for idx in range(num_oscillations):
            plt.axvline(events_times[idx], linestyle='--', c='gray')
        plt.show()

    return num_oscillations, events_idx, events_times, duration_cycles


if __name__ == '__main__':

    # 1) load or generate traj data
    flag_load_test_traj = True
    if flag_load_test_traj:
        times = np.loadtxt('input' + os.sep + 'traj_times.txt')
        r = np.loadtxt('input' + os.sep + 'traj_x.txt')
        r_choice = r[:, 1]  # try idx 0 or 1 (active/total cyclin)
    else:
        times = np.linspace(0, 5.2, 1000)
        r_choice = np.sin(2 * np.pi * times - 0)

    # 2) main detection call
    xlow, xhigh = 1, 2
    state_choice = 0
    #num_oscillations, events_idx, events_times, duration_cycles = detect_oscillations_scipy(
    #    times, r_choice, show=Tru, state_choice=state_choice)
    num_oscillations, events_idx, events_times, duration_cycles = detect_oscillations_manual(
        times, r_choice, xlow=xlow, xhigh=xhigh, show=True, state_choice=state_choice)

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
        #num_oscillations, events_idx, events_times, duration_cycles = detect_oscillations_scipy(times, r_choice, show=True, state_choice=state_choice)
        num_oscillations, events_idx, events_times, duration_cycles = detect_oscillations_manual(
            times, r_choice, xlow=xlow, xhigh=xhigh, show=True, state_choice=state_choice)

        print('\nTimeseries has %d oscillations' % num_oscillations)
        print('Oscillation info:')
        for idx in range(num_oscillations):
            print('\t(%d of %d) - Index of event: %d:' % (idx, num_oscillations, events_idx[idx]))
            print('\t(%d of %d) - Time of event: %.2f:' % (idx, num_oscillations, events_times[idx]))
            print('\t(%d of %d) - Period of cycle: %.2f' % (idx, num_oscillations, duration_cycles[idx]))
