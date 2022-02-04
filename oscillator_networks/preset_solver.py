""" options for solve_ivp

Reference:

solver_kwargs['t_eval'] = None  # None or np.linspace(0, 50, 2000)  np.linspace(15, 50, 2000)
solver_kwargs['max_step'] = np.Inf  # try 1e-1 or 1e-2 if division time-sequence is buggy as a result of large adaptive steps
"""


PRESET_SOLVER = {
    'solve_ivp_radau_default': dict(
    ),
    'solve_ivp_radau_minstep': dict(
        min_step=1e-1
    )
}
