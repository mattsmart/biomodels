""" options for solve_ivp

Reference:

solver_kwargs['t_eval'] = None  # None or np.linspace(0, 50, 2000)  np.linspace(15, 50, 2000)
solver_kwargs['max_step'] = np.Inf  # try 1e-1 or 1e-2 if division time-sequence is buggy as a result of large adaptive steps
"""


PRESET_SOLVER = {}
PRESET_SOLVER['solve_ivp_radau_default'] = dict(
    label='solve_ivp_radau_default',
    dynamics_method='solve_ivp',
    kwargs=dict(method='Radau'),
)

PRESET_SOLVER['solve_ivp_radau_minstep'] = dict(
    label='solve_ivp_radau_minstep',
    dynamics_method='solve_ivp',
    kwargs=dict(method='Radau', min_step=1e-1),
)

PRESET_SOLVER['solve_ivp_radau_relaxed'] = dict(
    label='solve_ivp_radau_minstep',
    dynamics_method='solve_ivp',
    kwargs=dict(method='Radau', atol=1e-5),
)
