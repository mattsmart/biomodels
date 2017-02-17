#from constants import *  # TODO fix constants.py first
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from sympy import plot_implicit, symbols, Eq
from sympy.plotting import plot as symplt


# specify setup
N = 3  # num types ligands
M = 1  # num types receptors
K = 1  # num types output molecules
assert(N == 2 or N == 3)  # for plotting, though maybe use colourmap for 4d

print "Settings: N = %d, M = %d, K = %d\n%d Ligand types\n%d Receptor types\n%d Response molecules\n" % (N,M,K,N,M,K)


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
    A1_fixed = 5.0
    A2_fixed = 5.0
    gamma_fixed = 9.9
    h11 = 1.0  
    h21 = 0.0
    h12 = 0.0
    h22 = 0.5
    s0 = 1.0
    
    A_fixed = np.array([[A1_fixed],[A2_fixed]])
    W_fixed = np.array([[gamma_fixed]])
    H_fixed = np.array([[h11, h21], [h12, h22]])  # Note diagonalized H corresponds to no crosstalk
    s_fixed = np.array([[s0]])
    A, W, H, s = get_parameters(N,M,K, A=A_fixed, W=W_fixed, H=H_fixed, s=s_fixed)
    
    L1, L2 = symbols('L1 L2')
    C_tilde = (A[0,0] + A[1,0] - W[0,0]*s[0,0])
    x1 = H[0,0]*L1 + H[1,0]*L2  # TODO: maybe there is nice linear way to extend this notation
    x2 = H[0,1]*L1 + H[1,1]*L2
    goveq = Eq((A[0,0]-W[0,0]*s[0,0])*x1 + (A[1,0]-W[0,0]*s[0,0])*x2 + x1*x2*C_tilde, W[0,0]*s[0,0])

    center_fixed = [-(A[1,0] - W[0,0]*s[0,0]) / (C_tilde*H[0,0]), -(A[0,0] - W[0,0]*s[0,0]) / (C_tilde * H[1,1])]
    print "center at", center_fixed, "if H is diagonal"

    extent = 1000
    plt_title = 'Ligand concentrations satisfying s=%.2f (N=%d, M=%d, K=%d)' % (s[0],N,M,K)
    plt_save = 'l1vsl2_%d%d%d' % (N,M,K)
    #p1 = plot_implicit(goveq, (L1, -extent, extent), (L2, -extent, extent), adaptive=False)
    #p1 = plot_implicit(goveq, (L1, -extent, extent), (L2, -extent, extent), adaptive=False, points=400)
    p1 = plot_implicit(goveq, (L1, -extent, extent), (L2, -extent, extent), depth=2, title=plt_title, xlabel='L1', ylabel='L2')
    mplplt = p1._backend
    fig = mplplt.fig
    ax = fig.gca()
    note = 'A1 = %.2f \nA2 = %.2f \ngamma = %.2f \nh11 = %.2f, h21 = %.2f\nh12 = %.2f, h22 = %.2f' % (A[0,0], A[1,0], W[0,0], H[0,0], H[0,1], H[1,0], H[1,1])
    ax.text(p1.xlim[1]*0.55, p1.ylim[1]*0.6, note, fontsize=9)
    ax.plot(center_fixed[0], center_fixed[1],'o')
    fig.savefig(plt_save + '.pdf')


if N == 3 and M == 1 and K == 1:
    #expect plane in this case
    #can plot with sympy 3d parametric plot function using parametric equations for plane
    #can plot using matplotlib plot_surface and z=z(x,y) to get height
    A, W, H, s = get_parameters(N,M,K)

    # setup variables
    gamma = W[0,0]
    s1 = s[0,0]
    A1 = A[0,0]
    h1, h2, h3 = H[0,0:3]
    normal = [h1,h2,h3]
    C = gamma*s1/(A1-gamma*s1)
    intercepts = [(C/h1,0,0), (0,C/h2,0), (0,0,C/h3)]
    
    # create surface
    L1range = np.linspace(0.0, C/h1, 100)
    L2range = np.linspace(0.0, C/h2, 100)
    print L1range
    xx, yy = np.meshgrid(L1range, L2range)
    z = (C - normal[0]*xx - normal[1]*yy) * 1. /normal[2]

    # plot surface
    plt_title = 'Ligand concentrations satisfying s=%.2f (N=%d, M=%d, K=%d)' % (s[0],N,M,K)
    plt_save = 'l1vsl2vsl3_%d%d%d' % (N,M,K)
    cmap = colors.ListedColormap(['white', 'red'])
    bounds=[0,5,10]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(xx, yy, z, alpha=0.4,  cmap=cmap, color='blue')
    ax.scatter(intercepts[0] , intercepts[1] , intercepts[2],  color=['red','green','blue'])
    ax.set_zlim(0.0, intercepts[2][2])
    ax.set_xlabel('L1')
    ax.set_ylabel('L2')
    ax.set_zlabel('L3')
    
    note = 'A1 = %.2f \ngamma = %.2f \nh1 = %.2f, h2 = %.2f, h3 = %.2f' % (A[0,0], W[0,0], H[0,0], H[0,1], H[0,2])
    ax.text(intercepts[0][0]*0.55, intercepts[1][1]*0.6, intercepts[2][2]*0.6, note, fontsize=7)

    #ax.view_init(-45, -15)
    ax.view_init(5, 35)
    plt.show()
    fig.savefig(plt_save + '.pdf')


if N == 3 and M == 2 and K == 2:
    A, W, H, s = get_parameters(N,M,K)
    print "not implemented"
