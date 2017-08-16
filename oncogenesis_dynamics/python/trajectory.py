import matplotlib.pyplot as plt
import numpy as np
from os import sep

from constants import OUTPUT_DIR
from formulae import fp_location_general
from plotting import plot_simplex

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

# params
alpha_plus = 0.05
alpha_minus = 4.95
mu = 0.77
a = 1.0
b = 8.369856428  #1.376666
c = 2.6
N = 100.0
v_x = 1.0
v_y = 0.0
v_z = 0.0
delta = 1 - b
s = c - 1
params = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z]

# =====================
# FIGURE SETUP
# =====================
fig = plot_simplex(N)
ax = fig.gca()
plt_title = 'Trajectory'
plt_save = 'trajectory'

# =====================
# SIMULATE
# =====================
init_cond = [95.0, 5.0, 0.0]
#x0 = [0.2, 0.0, 99.8]
#x0 = [0.2, 99.8, 0.0]

#dt = 0.005  #dt = 0.0001
#times = np.arange(0,100, dt)
dt = 0.05  #dt = 0.0001
times = np.arange(0,8000, dt)
r = np.zeros((len(times), 3))
r[0] = np.array(init_cond)

print "Trajectory loop:"
for idx, t in enumerate(times[:-1]):
    x,y,z = r[idx]
    fbar = (a*x + b*y + c*z + v_x + v_y + v_z) / N
    v = np.array([v_x - x*alpha_plus + y*alpha_minus        + (a - fbar)*x,
                  v_y + x*alpha_plus - y*(alpha_minus + mu) + (b - fbar)*y,
                  v_z +                y*mu                 + (c - fbar)*z])
    r[idx+1] = r[idx] + v*dt
    if idx % 1000 == 0:
        print r[idx+1], t
print 'Done trajectory'

# =====================
# FP COMPARISON
# =====================
if v_x == 0 and v_y == 0 and v_z == 0:
    solver_numeric = False
else:
    solver_numeric = True
predicted_fps = fp_location_general(params, solver_numeric=solver_numeric, solver_fast=False)
print "Predicted FPs:"
print predicted_fps[0]
print predicted_fps[1]
print predicted_fps[2]

# =====================
# PLOT SHOW
# =====================
#ax.view_init(-45, -15)
ax.view_init(5, 35)
ax.plot(r[:,0], r[:,1], r[:,2], label='trajectory')
#ax.plot([x1[0]], [x1[1]], [x1[2]], label='x_weird')
#ax.legend()
plt.show()
fig.savefig(OUTPUT_DIR + sep + plt_save + '.png')
