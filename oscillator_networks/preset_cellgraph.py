import numpy as np


PRESET_CELLGRAPH = {
    'PWL3_swap_copy': dict(
        num_cells=1,
        style_ode='PWL3_swap',
        style_detection='manual_crossings',
        style_division='copy',
        t0=0,
        t1=65,
        state_history=np.array([[0, 0, 0]]).T,
        verbosity=0,
        mods_params_ode={}
    )
}

# Variant of 'PWL3_swap_copy'
PRESET_CELLGRAPH['PWL3_swap_partition_ndiv_bam'] = PRESET_CELLGRAPH['PWL3_swap_copy'].copy()
PRESET_CELLGRAPH['PWL3_swap_partition_ndiv_bam']['style_division'] = 'partition_ndiv_bam'

PRESET_CELLGRAPH['PWL3_swap_partition_ndiv_all'] = PRESET_CELLGRAPH['PWL3_swap_copy'].copy()
PRESET_CELLGRAPH['PWL3_swap_partition_ndiv_all']['style_division'] = 'partition_ndiv_all'
