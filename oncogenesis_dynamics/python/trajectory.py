import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from sympy import plot_implicit, symbols, Eq
from sympy.plotting import plot as symplt
from sympy.plotting import plot3d_parametric_line

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


# constants
alpha_plus = 0.4
alpha_minus = 0.5
mu = 0.01
a = 1.0
b = 1.01*1.376666 #1.376666
delta = 1-b
c = 1.2
s = c - 1
N = 100
v_x = 0
v_y = 0
v_z = 0

# =====================
# FIND POSSIBLE FP
# =====================
def q_get(sign):
    #sign: must be +1 or -1
    assert sign in [-1,+1]
    bterm = alpha_plus - alpha_minus - mu - delta
    return 0.5/alpha_minus * (bterm + sign*np.sqrt(bterm**2 + 4*alpha_minus*alpha_plus))
                              
def xvec_get(q):
    xi = N*(s + alpha_plus - alpha_minus*q) / (s + (delta + s)*q)
    yi = q*xi
    zi = N - xi - yi
    return xi, yi, zi

x1_fp = xvec_get(q_get(+1))
x2_fp = xvec_get(q_get(-1))
print "possible FPs"
print x1_fp
print x2_fp


# =====================
# FIGURE SETUP
# =====================
normal = [1,1,1]
intercepts = [(N,0,0), (0,N,0), (0,0,N)]
# create surface
x1range = np.linspace(0.0, N, 100)
x2range = np.linspace(0.0, N, 100)
xx, yy = np.meshgrid(x1range, x2range)
z = (N - normal[0]*xx - normal[1]*yy) * 1. /normal[2]

# plot surface
plt_title = 'Model B trajectory'
plt_save = 'model_B_trajectory'
cmap = colors.ListedColormap(['white', 'red'])
bounds=[0,5,10]
norm = colors.BoundaryNorm(bounds, cmap.N)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(xx, yy, z, alpha=0.4,  cmap=cmap, color='blue')
ax.scatter(intercepts[0] , intercepts[1] , intercepts[2],  color=['red','green','blue'])
ax.set_title(plt_title)
ax.set_zlim(0.0, intercepts[2][2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

#note = 'A1 = %.2f \ngamma = %.2f \nh1 = %.2f, h2 = %.2f, h3 = %.2f' % (A[0,0], W[0,0], H[0,0], H[0,1], H[0,2])
#ax.text(intercepts[0][0]*0.55, intercepts[1][1]*0.6, intercepts[2][2]*0.6, note, fontsize=7)

# =====================
# SIMULATE
# =====================

x0 = [95.0, 5.0, 0.0]
#x0 = [0.2, 0.0, 99.8]
# weird fixed point from octave
x1 = [31.3862, 63.2379, 5.3759]

#dt = 0.005  #dt = 0.0001
#times = np.arange(0,100, dt)
dt = 0.05  #dt = 0.0001
times = np.arange(0,4000, dt)
r = np.zeros((len(times), 3))
r[0] = np.array(x0)

for idx, t in enumerate(times[:-1]):
    x,y,z = r[idx]
    fbar = (a*x + b*y + c*z + v_x + v_y + v_z) / N
    v = np.array([v_x - x*alpha_plus + y*alpha_minus        + (a - fbar)*x,
                  v_y + x*alpha_plus - y*(alpha_minus + mu) + (b - fbar)*y,
                  v_z +                y*mu                 + (c - fbar)*z])
    r[idx+1] = r[idx] + v*dt

    # Draw lines colored by speed
    #c = clip( [np.linalg.norm(v) * 0.005], 0, 1 )[0]
    #lorenz.append( pos=r[idx+1], color=(c,0, 1-c) )
    #rate( 500 )

    if idx % 1000 == 0:
        print r[idx+1], t

print 'done computing'

# =====================
# PLOT SHOW
# =====================

#ax.view_init(-45, -15)

ax.view_init(5, 35)
ax.plot(r[:,0], r[:,1], r[:,2], label='trajectory')
x_qss = 32.366645541667353
#ax.scatter([x_qss], [100-x_qss], [0], color='red')
ax.plot([x1[0]], [x1[1]], [x1[2]], label='x_weird')
#ax.legend()
plt.show()

fig.savefig(plt_save + '.pdf')
