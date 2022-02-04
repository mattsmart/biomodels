import numpy as np

from preset_cellgraph import PRESET_CELLGRAPH
from preset_solver import PRESET_SOLVER


SWEEP_SOLVER = PRESET_SOLVER['solve_ivp_radau_default']
SWEEP_BASE_CELLGRAPH = PRESET_CELLGRAPH['PWL3_swap_copy']

PRESET_SWEEP = {
    '1d_epsilon': dict(
        sweep_label='preset_1d_epsilon',
        base_cellgraph_kwargs=SWEEP_BASE_CELLGRAPH,
        params_name='epsilon',
        params_values=np.linspace(0.01, 0.3, 20),
        params_variety=['sc_ode'],
        solver_kwargs=SWEEP_SOLVER
    )
}
