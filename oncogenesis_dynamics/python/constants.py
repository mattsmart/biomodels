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
                                  params[10] -> mu_base    (typically 0)
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
             9: "v_z",
             10: "mu_base"}
PARAMS_ID_INV = {v: k for k, v in PARAMS_ID.iteritems()}
STATES_ID = {0: "x", 1: "y", 2: "z"}
STATES_ID_INV = {v: k for k, v in STATES_ID.iteritems()}
BIFURC_DICT = {i: "bifurc_" + PARAMS_ID[i] for i in xrange(len(PARAMS_ID))}
VALID_BIFURC_PARAMS = ["bifurc_b", "bifurc_c", "bifurc_mu"]  # list of implemented bifurcation parameters
SIM_METHODS = ["euler", "rk4", "libcall", "gillespie"]
ODE_SYSTEMS = ["default", "feedback_z", "feedback_yz", "feedback_mu_XZ_model"]

OUTPUT_DIR = "output"

CSV_HEADINGS = ['bifurc_id', 'x0_x', 'x0_y', 'x0_z', 'x0_stab', 'x1_x', 'x1_y', 'x1_z', 'x1_stab', 'x2_x', 'x2_y',
                  'x2_z', 'x2_stab']
CSV_DATA_TYPES = {'bifurc_b': float,
                  'x0_x': float, 'x0_y': float, 'x0_z': float, 'x0_stab': bool,
                  'x1_x': float, 'x1_y': float, 'x1_z': float, 'x1_stab': bool,
                  'x2_x': float, 'x2_y': float, 'x2_z': float, 'x2_stab': bool}

SIM_METHOD = "libcall"  # standard ODE integration method
INIT_COND = [99.9, 0.1, 0.0]  # default initial cond for trajectory
NUM_TRAJ = 200  # number of trajectories for trajectory_multiple
TIME_START = 0.0  # standard trajectory start time
TIME_END = 16000.0  # standard trajectory end time
NUM_STEPS = 20000  # number of time steps in each trajectory (non-gillespie)

PARAM_Z0_RATIO = 0.1           # size of z0 relative to N (in feedback_z function)
PARAM_Y0_PLUS_Z0_RATIO = 0.1   # size of y0 + z0 relative to N (in feedback_yz function)
PARAM_GAMMA = 9.0              # when z->N, mu_base-> approx (K + 1)*mu_base
PARAM_HILL = 1                 # (unused currently) hill parameter 'n'

# PLOTTING
# simplex bifurcation diagram colours
X0_COL = ["grey", "black"]  # black stable (grey unstable)
X1_COL = ["red", "blue"]  # blue stable (red unstable)
X2_COL = ["green", "magenta"]  # magenta stable (green unstable)
# bifurcation diagram colours for x*=(x,y,z) z fixed points in RGB
DEFAULT_X_COLOUR = (169/255.0, 209/255.0, 142/255.0)
DEFAULT_Y_COLOUR = (255/255.0, 217/255.0, 102/255.0)
DEFAULT_Z_COLOUR = (244/255.0, 177/255.0, 131/255.0)
