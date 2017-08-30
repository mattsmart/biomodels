"""
Comments
- current implementation for bifurcation along VALID_BIFURCATION_PARAMS only
- no stability calculation implemented (see matlab code for that)

Conventions
- params is 7-vector of the form: params[0] -> alpha_plus
                                  params[1] -> alpha_minus
                                  params[2] -> mu
                                  params[3] -> a           (usually normalized to 1)
                                  params[4] -> b           (b = 1 - delta)
                                  params[5] -> c           (c = 1 + s)
                                  params[6] -> N           (float not int)
                                  params[7] -> v_x
                                  params[8] -> v_y         (typically 0)
                                  params[9] -> v_z         (typically 0)
- if an element of params is specified as None then a bifurcation range will be be found and used
"""

PARAMS_ID = {0: "alpha_plus",
             1: "alpha_minus",
             2: "mu",
             3: "a",
             4: "b",
             5: "c",
             6: "N",
             7: "v_x",
             8: "v_y",
             9: "v_z"}
PARAMS_ID_INV = {v: k for k, v in PARAMS_ID.iteritems()}
STATES_ID = {0: "x", 1: "y", 2: "z"}
STATES_ID_INV = {v: k for k, v in STATES_ID.iteritems()}
BIFURC_DICT = {i: "bifurc_" + PARAMS_ID[i] for i in xrange(len(PARAMS_ID))}
VALID_BIFURC_PARAMS = ["bifurc_b", "bifurc_c"]  # list of implemented bifurcation parameters
ODE_METHODS = ["euler", "rk4", "libcall"]
ODE_SYSTEMS = ["default", "feedback"]

OUTPUT_DIR = "output"

X0_COL = ["grey", "black"]  # black stable (grey unstable)
X1_COL = ["red", "blue"]  # blue stable (red unstable)
X2_COL = ["green", "magenta"]  # magenta stable (green unstable)

CSV_HEADINGS = ['bifurc_id', 'x0_x', 'x0_y', 'x0_z', 'x0_stab', 'x1_x', 'x1_y', 'x1_z', 'x1_stab', 'x2_x', 'x2_y',
                  'x2_z', 'x2_stab']
CSV_DATA_TYPES = {'bifurc_b': float,
                  'x0_x': float, 'x0_y': float, 'x0_z': float, 'x0_stab': bool,
                  'x1_x': float, 'x1_y': float, 'x1_z': float, 'x1_stab': bool,
                  'x2_x': float, 'x2_y': float, 'x2_z': float, 'x2_stab': bool}

PARAM_Z0_RATIO = 0.5  # size of z0 relative to N (in feedback function)
PARAM_HILL = 2        # (unused currently) hill parameter 'n'
