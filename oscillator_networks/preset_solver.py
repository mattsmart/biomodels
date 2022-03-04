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
    label='solve_ivp_radau_relaxed',
    dynamics_method='solve_ivp',
    kwargs=dict(method='Radau', atol=1e-5),
)

PRESET_SOLVER['solve_ivp_BDF_default'] = dict(
    label='solve_ivp_BDF_default',
    dynamics_method='solve_ivp',
    kwargs=dict(method='BDF'),
)

PRESET_SOLVER['solve_ivp_LSODA_default'] = dict(
    label='solve_ivp_LSODA_default',
    dynamics_method='solve_ivp',
    kwargs=dict(method='LSODA'),
)


# TODO - not working
PRESET_SOLVER['diffeqpy_default'] = dict(
    label='diffeqpy_default',
    dynamics_method='diffeqpy',
    kwargs=dict(abstol=1e-8, reltol=1e-4),  # assumes RadauIIA5 solver for now
)

# TODO - not working; also try to numba our other functions if possible?
PRESET_SOLVER['numba_lsoda'] = dict(
    label='numba_lsoda',
    dynamics_method='numba_lsoda',
    kwargs=dict(atol=1e-8, rtol=1e-4),
)