#from constants import *  # TODO fix constants.py first
import matplotlib.pyplot as plt
import numpy as np
from sympy import plot_implicit, symbols, Eq
from sympy.plotting import plot as symplt


# specify setup
N = 2  # num types ligands
M = 2  # num types receptors
K = 1  # num types output molecules
assert(N == 2 or N == 3)  # for plotting

# specify dynamics (all uniform [0,1) sampling)
A = np.random.rand(M,K) * 10  # want Aij > Wij
W_notdiag = np.random.rand(K,K)
W = np.diag(np.diag(W_notdiag))
H = np.random.rand(M,N)

# specify output molecule concentrations
s = np.random.rand(K,1)

# compute space of associated ligand concentrations
# plot this region
# TODO automate this
if N == 2 and M == 1 and K == 1:
    # compute
    C = W[0,0]*s[0,0] / (A[0,0] - W[0,0]*s[0,0])
    def get_L2(L1):
        return (C - H[0,0]*L1)/H[0,1]
    L2_intercept = C/H[0,0]
    L1 = np.linspace(0.0,L2_intercept,100)
    L2 = [get_L2(a) for a in L1]
    # plot
    plt.plot(L1,L2)
    fig = plt.gcf()
    ax = plt.gca()
    plt.title('Ligand concentrations satisfying s=%.2f (N=%d, M=%d, K=%d)' % (s[0],N,M,K))
    note = 'A = %.2f \ngamma = %.2f \nh11 = %.2f, h21 = %.2f' % (A[0,0], H[0,0], H[0,1], W[0,0])
    ax.text(max(L1)*0.65, max(L2)*0.8, note, fontsize=11)
    plt.xlabel('[L1]', fontsize=12)
    plt.ylabel('[L2]', fontsize=12)
    plt.show()
    fig.savefig('l1vsl2_%d%d%d.pdf' % (N,M,K))

if N == 2 and M == 2 and K == 1:
    """
    # compute
    C = W[0,0]*s[0,0] / (A[0,0] - W[0,0]*s[0,0])
    def get_L2(L1):
        return (C - H[0,0]*L1)/H[0,1]
    L2_intercept = C/H[0,0]
    L1 = np.linspace(0.0,L2_intercept,100)
    L2 = [get_L2(a) for a in L1]
    # plot
    plt.plot(L1,L2)
    fig = plt.gcf()
    ax = plt.gca()
    plt.title('Ligand concentrations satisfying s=%.2f (N=%d, M=%d, K=%d)' % (s[0],N,M,K))
    note = 'A = %.2f \ngamma = %.2f \nh11 = %.2f, h21 = %.2f' % (A[0,0], H[0,0], H[0,1], W[0,0])
    ax.text(max(L1)*0.65, max(L2)*0.8, note, fontsize=11)
    plt.xlabel('[L1]', fontsize=12)
    plt.ylabel('[L2]', fontsize=12)
    plt.show()
    fig.savefig('l1vsl2_%d%d%d.pdf' % (N,M,K))
    """
    L1, L2 = symbols('L1 L2')
    gamma = W[0,0]
    a = H[0,0]*H[1,0]
    b = H[0,0]*H[1,1] + H[1,0]*H[0,1]
    c = H[1,1]*H[0,1]
    A_tilde = A[0,0] - A[1,0]
    C_tilde = (A[0,0] + A[1,0] - gamma*s[0])[0]
    h1_tilde = H[0,0] + H[1,0]
    h2_tilde = H[0,1] + H[1,1]
    goveq = Eq(a * C_tilde * (L1**2) \
               + L1*(b*C_tilde*L2 + A_tilde*H[1,0] + (A[0,0] - gamma*s[0])*h1_tilde) \
               + c*C_tilde*L2 - gamma*s[0] - L2*((gamma*s[0]-A[0,0])*h2_tilde + A_tilde*H[1,1]) \
               , 5)
    p1 = plot_implicit(goveq, (L1, -4, 4), (L2, -4, 4), adaptive=False)
    #p3 = plot_implicit(Eq(x**2 + y**2, 5), (x, -4, 4), (y, -4, 4), depth = 2)
    #p4 = plot_implicit(Eq(x**2 + y**2, 5), (x, -5, 5), (y, -2, 2), adaptive=False)
    #p5 = plot_implicit(Eq(x**2 + y**2, 5), (x, -5, 5), (y, -2, 2), adaptive=False, points=400)

# TODO: check goveq form not right.. check hyperbola to ellipse condition.. cleanup plot
