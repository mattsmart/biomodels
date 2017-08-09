import numpy as np
#from scipy.optimize import fsolve
from sympy import Symbol, solve

from formulae import fp_location, q_get

# DYNAMICS PARAMETERS
alpha_plus = 0.05 #0.4
alpha_minus = 4.95 #0.5
mu = 0.77 #0.77 #0.01
a = 1.0
b = 1.1
c = 2.6 #1.2
N = 100.0
v_x = 0.0
v_y = 0.0
v_z = 0.0
if b is not None:
    delta = 1 - b
if c is not None:
    s = c - 1
params = [alpha_plus, alpha_minus, mu, a, b, c, N, v_x, v_y, v_z]

sym_x = Symbol("x")
sym_y = Symbol("y")
sym_z = Symbol("z")

xdot = (c-a)/N*sym_x**2 + (c-b)/N*sym_x*sym_y + (a-c-alpha_plus-(v_x+v_y+v_z)/N)*sym_x + alpha_minus*sym_y + v_x
ydot = (c-b)/N*sym_y**2 + (c-a)/N*sym_x*sym_y + (b-c-alpha_minus-mu-(v_x+v_y+v_z)/N)*sym_y + alpha_plus*sym_x + v_y
zz = N - sym_x - sym_y - sym_z

eqns = (xdot, ydot, zz)

solution = solve(eqns)
print solution

orderdict = {0:sym_x, 1:sym_y, 2:sym_z}
xA = [float(solution[0][orderdict[i]]) for i in xrange(3)]
xB = [float(solution[1][orderdict[i]]) for i in xrange(3)]
xC = [float(solution[2][orderdict[i]]) for i in xrange(3)]

# test validity 1
q1 = q_get(params, 1)
q2 = q_get(params, -1)
print fp_location(params, q1)
print fp_location(params, q2)

# test validity 2
for fp in [xA, xB, xC]:
    x,y,z = fp
    fbar = (a * x + b * y + c * z + v_x + v_y + v_z) / N
    v = np.array([v_x - x * alpha_plus + y * alpha_minus + (a - fbar) * x,
                  v_y + x * alpha_plus - y * (alpha_minus + mu) + (b - fbar) * y,
                  v_z + y * mu + (c - fbar) * z])
    print fp, v

print "\nNow Jacobian"
M = np.array([[a-alpha_plus, alpha_minus,      0],
              [alpha_plus,   b-alpha_minus-mu, 0],
              [0,            mu,               c]])
# compute eigenvalues of 2x2 jacobian
def jacobian1(fp):
    print fp
    x, y, z = fp
    print "ANDDDDDDDDD", x, y, z
    r1 = [a-alpha_plus-c-(v_x+v_y+v_z)/N - 1/N*(2*(a-c)*x + (b-c)*y),
          alpha_minus - 1/N*(b-c)*x]
    r2 = [alpha_plus - 1/N*(a-c)*y,
          b-alpha_minus-c-mu-(v_x+v_y+v_z)/N - 1/N*((a-c)*x + 2*(b-c)*y)]
    return np.array([r1,r2])

def jacobian3d(fp):
    x, y, z = fp
    diag = a*x + b*y + c*z + v_x + v_y + v_z
    r1 = [diag + x*a, x*b, x*c]
    r2 = [y*a, diag + y*b, y*c]
    r3 = [z*a, z*b, diag + z*c]
    return M - 1/N*np.array([r1,r2,r3])

# do like in matlab script
"""
def jacobian2(fp):
    x, y, z = fp
    return M - 1/N*(avec)
"""
print jacobian1(xA)
print jacobian1(xB)
print jacobian1(xC)

"""
print "test3check"
print a-alpha_plus-c-(v_x+v_y+v_z)/N - 1/N*(2*(a-c)*0 + (b-c)*0)
print a-alpha_plus-c-(v_x+v_y+v_z)/N - 1/N*(2*(a-c)*99.5385531016556 + (b-c)*0.886507271481506)
print a-alpha_plus-c-(v_x+v_y+v_z)/N - 1/N*(2*(a-c)*-7175.25283881594 + (b-c)*8137.87539749042)
print "test3check 222"
print 1/N*(2*(a-c)*0 + (b-c)*0)
print 1/N*(2*(a-c)*99.5385531016556 + (b-c)*0.886507271481506)
print 1/N*(2*(a-c)*-7175.25283881594 + (b-c)*8137.87539749042)
"""

print "3D\n"
print "xA", xA
print "xB", xB
print "xC", xC
print "jacobs"
print "jxa", jacobian3d(xA), "\n"
print "jxb", jacobian3d(xB), "\n"
print "jxc", jacobian3d(xC), "\n"

V, D = np.linalg.eig(jacobian3d(xA))
print V
print D
