import numpy as np

from preset_cellgraph import PRESET_CELLGRAPH
from preset_solver import PRESET_SOLVER


SWEEP_SOLVER = PRESET_SOLVER['solve_ivp_radau_default']['kwargs']
SWEEP_BASE_CELLGRAPH = PRESET_CELLGRAPH['PWL3_swap_copy']

PRESET_SWEEP = {}
PRESET_SWEEP['1d_epsilon_copy'] = dict(
	sweep_label='sweep_preset_1d_epsilon',
	base_cellgraph_kwargs=SWEEP_BASE_CELLGRAPH,
	params_name=[
		'epsilon'
	],
	params_values=[
		np.linspace(0.01, 0.4, 101)
	],
	params_variety=[
		'sc_ode'
	],
	solver_kwargs=SWEEP_SOLVER
)

# Variants of '1d_epsilon_copy'
PRESET_SWEEP['1d_epsilon_ndiv_bam'] = PRESET_SWEEP['1d_epsilon_copy'].copy()
PRESET_SWEEP['1d_epsilon_ndiv_bam']['base_cellgraph_kwargs'] = PRESET_CELLGRAPH['PWL3_swap_partition_ndiv_bam']

PRESET_SWEEP['1d_epsilon_ndiv_all'] = PRESET_SWEEP['1d_epsilon_copy'].copy()
PRESET_SWEEP['1d_epsilon_ndiv_all']['base_cellgraph_kwargs'] = PRESET_CELLGRAPH['PWL3_swap_partition_ndiv_all']


PRESET_SWEEP['1d_diffusion_copy'] = dict(
	sweep_label='sweep_preset_1d_diffusion',
	base_cellgraph_kwargs=SWEEP_BASE_CELLGRAPH,
	params_name=[
		'diffusion_rate'
	],
	params_values=[
		np.linspace(0.0, 10, 101)
	],
	params_variety=[
		'meta_cellgraph'
	],
	solver_kwargs=SWEEP_SOLVER
)

# Variants of '1d_diffusion_copy'
PRESET_SWEEP['1d_diffusion_ndiv_bam'] = PRESET_SWEEP['1d_diffusion_copy'].copy()
PRESET_SWEEP['1d_diffusion_ndiv_bam']['base_cellgraph_kwargs'] = PRESET_CELLGRAPH['PWL3_swap_partition_ndiv_bam']

PRESET_SWEEP['1d_diffusion_ndiv_all'] = PRESET_SWEEP['1d_diffusion_copy'].copy()
PRESET_SWEEP['1d_diffusion_ndiv_all']['base_cellgraph_kwargs'] = PRESET_CELLGRAPH['PWL3_swap_partition_ndiv_all']

# Variants of '2d_epsilon_diffusion_ndiv_bam'
PRESET_SWEEP['2d_epsilon_diffusion_ndiv_bam'] = dict(
	sweep_label='sweep_preset_1d_epsilon',
	base_cellgraph_kwargs=PRESET_CELLGRAPH['PWL3_swap_partition_ndiv_bam'],
	params_name=[
		'epsilon',
		'diffusion_rate'
	],
	params_values=[
		np.linspace(0.01, 0.3, 41),
		np.linspace(0.0, 10, 101)
	],
	params_variety=[
		'sc_ode',
		'meta_cellgraph'
	],
	solver_kwargs=SWEEP_SOLVER
)


# Variants of '2d_epsilon_diffusion_ndiv_bam'
PRESET_SWEEP['2d_epsilon_diffusion_ndiv_bam'] = dict(
	sweep_label='2d_epsilon_diffusion_ndiv_bam',
	base_cellgraph_kwargs=PRESET_CELLGRAPH['PWL3_swap_partition_ndiv_bam'],
	params_name=[
		'epsilon',
		'diffusion_rate'
	],
	params_values=[
		np.linspace(0.01, 0.3, 41),
		np.linspace(0.0, 8, 101)
	],
	params_variety=[
		'sc_ode',
		'meta_cellgraph'
	],
	solver_kwargs=SWEEP_SOLVER
)
