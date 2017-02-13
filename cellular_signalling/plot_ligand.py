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


def get_parameters(N, M, K, A=None, W=None, H=None, s=None):
    # default is random parameters with uniform [0,1) sampling
    if A is None:
        A = np.random.rand(M,K) * 10  # want Aij > Wij
    if W is None:
        W_notdiag = np.random.rand(K,K)
        W = np.diag(np.diag(W_notdiag))
    if H is None:
        H = np.random.rand(M,N)
    if s is None:
        # specify output molecule concentrations
        s = np.random.rand(K,1)        
    return A, W, H, s


# compute space of associated ligand concentrations
# plot this region
# TODO automate this
if N == 2 and M == 1 and K == 1:
    A, W, H, s = get_parameters(N,M,K)
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
    note = 'A = %.2f \ngamma = %.2f \nh11 = %.2f, h21 = %.2f' % (A[0,0], W[0,0], sH[0,0], H[0,1])
    ax.text(max(L1)*0.65, max(L2)*0.8, note, fontsize=11)
    plt.xlabel('[L1]', fontsize=12)
    plt.ylabel('[L2]', fontsize=12)
    plt.show()
    fig.savefig('l1vsl2_%d%d%d.pdf' % (N,M,K))

if N == 2 and M == 2 and K == 1:
    A_fixed = np.array([[5],[1]])
    W_fixed = np.array([[6]])
    s_fixed = np.array([[1]])
    A, W, H, s = get_parameters(N,M,K, A=A_fixed, W=W_fixed, s=s_fixed)
    
    L1, L2 = symbols('L1 L2')
    gamma = W[0,0]
    a = H[0,0]*H[1,0]
    b = H[0,0]*H[1,1] + H[1,0]*H[0,1]
    c = H[1,1]*H[0,1]
    A_tilde = A[0,0] - A[1,0]
    C_tilde = (A[0,0] + A[1,0] - gamma*s[0,0])
    h1_tilde = H[0,0] + H[1,0]
    h2_tilde = H[0,1] + H[1,1]
    x1 = H[0,0]*L1 + H[1,0]*L2  # TODO: maybe there is nice linear way to extend this notation
    x2 = H[0,1]*L1 + H[1,1]*L2
    goveq = Eq((A[0,0]-gamma*s[0,0])*x1 + (A[1,0]-gamma*s[0,0])*x2 + x1*x2*C_tilde, gamma*s[0,0])
    #goveq = Eq((A[0,0]-gamma*s[0,0])*(H[0,0]*L1 + H[1,0]*L2) + (A[1,0]-gamma*s[0,0])*(H[0,1]*L1 + H[1,1]*L2) + (H[0,0]*L1 + H[1,0]*L2)*(H[0,1]*L1 + H[1,1]*L2)*C_tilde, gamma*s[0,0])
    extent = 10
    plt_title = 'Ligand concentrations satisfying s=%.2f (N=%d, M=%d, K=%d)' % (s[0],N,M,K)
    plt_save = 'l1vsl2_%d%d%d' % (N,M,K)
    #p1 = plot_implicit(goveq, (L1, -extent, extent), (L2, -extent, extent), adaptive=False)
    #p1 = plot_implicit(goveq, (L1, -extent, extent), (L2, -extent, extent), adaptive=False, points=400)
    p1 = plot_implicit(goveq, (L1, -extent, extent), (L2, -extent, extent), depth=2, title=plt_title, xlabel='L1', ylabel='L2', save=plt_save +'_raw.pdf')
    mplplt = p1._backend
    fig = mplplt.fig
    ax = fig.gca()
    note = 'A1 = %.2f \nA2 = %.2f \ngamma = %.2f \nh11 = %.2f, h21 = %.2f\nh12 = %.2f, h22 = %.2f' % (A[0,0], A[1,0], W[0,0], H[0,0], H[0,1], H[1,0], H[1,1])
    ax.text(p1.xlim[1]*0.55, p1.ylim[1]*0.6, note, fontsize=9)
    #mplplt.show()
    fig.savefig(plt_save+'.pdf')
    #p1.title('Ligand concentrations satisfying s=%.2f (N=%d, M=%d, K=%d)' % (s[0],N,M,K))
    #fig.savefig('test.pdf')


# TODO: check goveq form not right.. check hyperbola to ellipse condition.. cleanup plot
