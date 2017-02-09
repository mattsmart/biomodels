#from constants import *  # TODO fix constants.py first
import matplotlib.pyplot as plt
import numpy as np


# specify setup
N = 2  # num types ligands
M = 1  # num types receptors
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

    
    


# plot
