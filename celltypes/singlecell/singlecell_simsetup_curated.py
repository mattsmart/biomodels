import numpy as np


N_FERRO = 6

curated = {
    'mutual_inhibition':
        {'XI': np.array([
            [1, -1],  # TF 1
            [-1, 1],  # TF 2
            [1, -1],  # identity gene
            [-1, 1],  # identity gene
            [1, 1],   # housekeeping gene (hardcoded)
            [1, 1]]),  # housekeeping gene (hardcoded)
            #    when gene 0 (col 1) is ON as in mem A, it promoted mem A and inhibits mem B
            #    when gene 1 (col 2) is ON as in mem B, it promoted mem A and inhibits mem B
          'W': np.array([
            [-5, 0, 0, 0, 0, 0],
            [0, -5, 0, 0, 0, 0],
            [-5, 5, 0, 0, 0, 0],
            [5, -5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]]),
          'celltype_labels': ['mem_A', 'mem_B'],
          'gene_labels': ['A_signal', 'B_signal', 'A_identity', 'B_identity', 'HK_1', 'HK_2']
         },
    'ferro':
        {'XI': np.ones((N_FERRO, 1)),
          'W': np.zeros((N_FERRO, N_FERRO)),
          'celltype_labels': [r'$\xi$'],
          'gene_labels': ['gene_%d' % idx for idx in range(N_FERRO)],
         },
    'ferroPerturb':
        {'XI': np.array([[1], [1], [1], [1], [2.5]]),
         'W': np.zeros((5, 5)),
         'celltype_labels': [r'$\xi$'],
         'gene_labels': ['gene_%d' % idx for idx in range(5)],
         },
    '3MemOrthog':
        {'XI': np.array([[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [ 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1],
                         [ 1, 1, 1,-1,-1,-1, 1, 1, 1,-1,-1,-1]]).T,
         'W': np.zeros((12, 12)),
         'celltype_labels': [r'$\xi_A$', r'$\xi_B$', r'$\xi_C$'],
         'gene_labels': ['gene_%d' % idx for idx in range(12)],
         },
    '3MemCorr':
        {'XI': np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1],
                         [1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1]]).T,
         'W': np.zeros((12, 12)),
         'celltype_labels': [r'$\xi_A$', r'$\xi_B$', r'$\xi_C$'],
         'gene_labels': ['gene_%d' % idx for idx in range(12)],
         },
    '3MemCorrPerturb':
        {'XI': np.array([[2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [4, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
                         [1, 2, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1]]).T,
         'W': np.zeros((12, 12)),
         'celltype_labels': [r'$\xi_A$', r'$\xi_B$', r'$\xi_C$'],
         'gene_labels': ['gene_%d' % idx for idx in range(12)],
         }
}

LABEL = 'ferro'
assert LABEL in list(curated.keys())

refine_W = True
random_W = True
if refine_W:
    # manually refine the W matrix of the chosen scheme
    Ntot = curated[LABEL]['XI'].shape[0]
    if random_W:
        W_0 = np.random.rand(Ntot, Ntot) * 2 - 1  # scale to Uniform [-1, 1]
        W_lower = np.tril(W_0, k=-1)
        W_diag = np.diag(np.diag(W_0))
        curated[LABEL]['W'] = (W_lower + W_lower.T + W_diag) / Ntot
    else:
        curated[LABEL]['W'][1, 1] = 10.0/Ntot
        curated[LABEL]['W'][2, 3] = -10.0/Ntot
        curated[LABEL]['W'][3, 2] = -10.0/Ntot

CURATED_XI = curated[LABEL]['XI']
CURATED_W = curated[LABEL]['W']
CURATED_CELLTYPE_LABELS = curated[LABEL]['celltype_labels']
CURATED_GENE_LABELS = curated[LABEL]['gene_labels']
