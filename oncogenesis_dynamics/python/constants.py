"""
Comments
- current implementation for bifurcation along VALID_BIFURCATION_PARAMS only
- no stability calculation implemented (see matlab code for that)

Conventions
- params is 13-vector of the form: params[0] -> alpha_plus
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
                                   params[11] -> c2         (typically 0)
                                   params[12] -> v_z2       (typically 0)
- if an element of params is specified as None then a bifurcation range will be be found and used
"""

# MODEL PARAMETERS AND LABELS
ODE_SYSTEMS = ["default", "feedback_z", "feedback_yz", "feedback_mu_XZ_model", "feedback_XYZZprime"]
FEEDBACK_SHAPES = ["constant", "hill", "step", "pwlinear"]
DEFAULT_FEEDBACK_SHAPE = "hill"
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
             10: "mu_base",
             11: "c2",
             12: "v_z2"}
PARAMS_ID_INV = {v: k for k, v in PARAMS_ID.iteritems()}
STATES_ID = {0: "x", 1: "y", 2: "z"}
STATES_ID_INV = {v: k for k, v in STATES_ID.iteritems()}

BIFURC_DICT = {i: "bifurc_" + PARAMS_ID[i] for i in xrange(len(PARAMS_ID))}
VALID_BIFURC_PARAMS = ["bifurc_b", "bifurc_c", "bifurc_mu"]  # list of implemented bifurcation parameters

# IO PARAMS
OUTPUT_DIR = "output"
CSV_HEADINGS = ['bifurc_id', 'x0_x', 'x0_y', 'x0_z', 'x0_stab', 'x1_x', 'x1_y', 'x1_z', 'x1_stab', 'x2_x', 'x2_y',
                  'x2_z', 'x2_stab']
CSV_DATA_TYPES = {'bifurc_b': float,
                  'x0_x': float, 'x0_y': float, 'x0_z': float, 'x0_stab': bool,
                  'x1_x': float, 'x1_y': float, 'x1_z': float, 'x1_stab': bool,
                  'x2_x': float, 'x2_y': float, 'x2_z': float, 'x2_stab': bool}

# ODE TRAJECTORY PARAMS
SIM_METHODS_VALID = ["euler", "rk4", "libcall", "gillespie"]
SIM_METHOD = "libcall"  # standard ODE integration method
INIT_COND = [99.9, 0.1, 0.0]  # default initial cond for trajectory
NUM_TRAJ = 200  # number of trajectories for trajectory_multiple
TIME_START = 0.0  # standard trajectory start time
TIME_END = 16000.0  # standard trajectory end time
NUM_STEPS = 20000  # number of time steps in each trajectory (non-gillespie)

# SYSTEM VARIANT PARAMS
HILL_Z0_RATIO = 0.1           # size of z0 relative to N (in feedback_z function)
HILL_Y0_PLUS_Z0_RATIO = 0.1   # size of y0 + z0 relative to N (in feedback_yz function)
PARAM_GAMMA = 9.0             # when z->N, mu_base-> approx (K + 1)*mu_base
HILL_EXP = 1                  # (unused currently) hill parameter 'n'

# COLOURS FOR PLOTTING
# simplex bifurcation diagram colours
X0_COL = ["grey", "black"]  # black stable (grey unstable)
X1_COL = ["red", "blue"]  # blue stable (red unstable)
X2_COL = ["green", "magenta"]  # magenta stable (green unstable)
# simplex basins from trajectories
BASIN_COLOUR_DICT = {0: 'blue', 1: 'red', 2: 'green'}
# bifurcation diagram colours for x*=(x,y,z) z fixed points in RGB
DEFAULT_X_COLOUR = (169/255.0, 209/255.0, 142/255.0)
DEFAULT_Y_COLOUR = (255/255.0, 217/255.0, 102/255.0)
DEFAULT_Z_COLOUR = (244/255.0, 177/255.0, 131/255.0)
# FPT histogram colours
GREY = (169 / 255.0, 169 / 255.0, 169 / 255.0)
BLUE = (119 / 255.0, 158 / 255.0, 203 / 255.0)
GREY_DARK = (149 / 255.0, 149 / 255.0, 149 / 255.0)
COLOUR_EPS = 20.0 / 255.0
X_DARK = [i - COLOUR_EPS for i in DEFAULT_X_COLOUR]
Z_DARK = [i - COLOUR_EPS for i in DEFAULT_Z_COLOUR]
COLOURS_GREY = [DEFAULT_X_COLOUR, GREY, DEFAULT_Z_COLOUR]
COLOURS_DARK_GREY = [X_DARK, GREY_DARK, Z_DARK]
COLOURS_DARK_BLUE = [X_DARK, BLUE, Z_DARK]
# bistability phase diagram colours
Y_LOW = tuple([0.5 * (DEFAULT_X_COLOUR[idx] + DEFAULT_Y_COLOUR[idx]) for idx in xrange(3)])
Y_HIGH = tuple([0.5 * (DEFAULT_Y_COLOUR[idx] + DEFAULT_Z_COLOUR[idx]) for idx in xrange(3)])
Z_TO_COLOUR_BISTABLE_WIDE = [(0.0, GREY),             (1e-9, DEFAULT_X_COLOUR),   (0.15, Y_LOW),
                             (0.5, DEFAULT_Y_COLOUR), (0.85, Y_HIGH),             (1.0, DEFAULT_Z_COLOUR)]
Z_TO_COLOUR_BISTABLE = [(0.0, GREY), (1e-9, DEFAULT_X_COLOUR), (0.5, DEFAULT_Y_COLOUR), (1.0, DEFAULT_Z_COLOUR)]
Z_TO_COLOUR_ORIG = [(0.0, DEFAULT_X_COLOUR), (0.5, DEFAULT_Y_COLOUR), (1.0, DEFAULT_Z_COLOUR)]
