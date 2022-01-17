import numpy as np


def set_params_ode(style_ode):
    if style_ode == 'Yang2013':
        # reference is Yang2013 Table S1
        p = {
            'k_synth': 1,  # nM / min
            'a_deg': 0.01,  # min^-1
            'b_deg': 0.04,  # min^-1
            'EC50_deg': 32,  # nM
            'n_deg': 17,  # unitless
            'a_Cdc25': 0.16,  # min^-1
            'b_Cdc25': 0.80,  # min^-1
            'EC50_Cdc25': 35,  # nM
            'n_Cdc25': 11,  # unitless
            'a_Wee1': 0.08,  # min^-1
            'b_Wee1': 0.40,  # min^-1
            'EC50_Wee1': 30,  # nM
            'n_Wee1': 3.5,  # unitless
        }
    else:
        print("Warning: style_ode %s is not supported by get_params_ODE()" % style_ode)
        p = {}
    # add any extra parameters that are generic
    p['Bam_activity'] = 1  # as indicated in SmallCellCluster review draft p7
    p['Bam_deg'] = 0  # degradation rate; arbitrary, try 0 or 1e-2 to 1e-4
    return p


def vectorfield_Yang2013(params, x, y, z=0, two_dim=True):
    """
    Args:
        params - dictionary of ODE parameters used by Yang2013
        x - array-like
        y - array-like
        z - array-like
    Returns:
        array like of shape [x, y] or [x, y, z] depending on two_dim flag
    """
    p = params

    # "f(x)" factor of the review - degradation
    # TODO care if x = 0 -- add case?
    r_degradation = (p['EC50_deg'] / x) ** p['n_deg']
    degradation = p['a_deg'] + p['b_deg'] / (1 + r_degradation)
    degradation_scaled = degradation / (1 + z / p['Bam_activity'])  # as in p7 of SmallCellCluster Review draft

    # "g(x)" factor of the review - activation by Cdc25
    # TODO care if x = 0 -- add case?
    r_activation = (p['EC50_Cdc25'] / x) ** p['n_Cdc25']
    activation = p['a_Cdc25'] + p['b_Cdc25'] / (1 + r_activation)

    # "k_i" factor of the review - de-activation by Wee1
    r_deactivation_inv = (x / p['EC50_Wee1']) ** p['n_Wee1']
    deactivation = p['a_Wee1'] + p['b_Wee1'] / (1 + r_deactivation_inv)

    dxdt = p['k_synth'] - degradation_scaled * x + activation * (y - x) - deactivation * x
    dydt = p['k_synth'] - degradation_scaled * y
    #dzdt = np.zeros_like(dxdt)
    dzdt = -p['Bam_deg'] * z

    if two_dim:
        out = [dxdt, dydt]
    else:
        out = [dxdt, dydt, dzdt]

    return out
