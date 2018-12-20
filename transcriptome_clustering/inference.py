import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, Ridge


def check_symmetric(arr, tol=1e-8):
    return np.allclose(arr, arr.T, atol=tol)


def vectorize_matrix(arr, order='C'):
    """
    Convert NxN matrix into N^2 1-dim array, row by row
        'C' - C-style, do row-by-row (default)
        'F' - Fortran-style, do column-by-column
    """
    assert all([len(arr.shape) == 2, arr.shape[0] == arr.shape[1], order in ['C', 'F']])
    vec = arr.flatten(order=order)
    return vec


def matrixify_vector(vec, order='C'):
    """
    Convert N^2xN^2 vector (1-dim array) into NxN matrix (2-dim array), row by row
        'C' - C-style, do row-by-row (default)
        'F' - Fortran-style, do column-by-column
    """
    assert len(vec.shape) == 1 and order in ['C', 'F']
    N_sqr = vec.shape[0]
    N = int(np.sqrt(N_sqr))
    assert N_sqr == N*N
    if order == 'C':
        vec = vec.reshape((N, N))
    else:
        vec = vec.reshape((N, N), order='F')
    return vec


def arr_cross_eye(arr):
    """
    Computes the block outer product (kronecker product) "A cross I" which will be of size N^2 x N^2
    For 2x2 arr, it looks like [[a11 * I, a12 * I], [a21 * I, a22 * I]]
    """
    assert len(arr.shape) == 2 and arr.shape[0] == arr.shape[1]
    N = arr.shape[0]
    eye = np.eye(N)
    tiled = np.zeros((N**2, N**2))
    for i in xrange(N):
        i_start = N * i
        i_end = N * (i + 1)
        for j in xrange(N):
            j_start = N * j
            j_end = N * (j + 1)
            tiled[i_start:i_end, j_start:j_end] = arr[i, j] * eye
    return tiled


def eye_cross_arr(arr):
    """
    Computes the block outer product (kronecker product) "I cross arr" which will be of size N^2 x N^2
    For 2x2 arr, it looks like [[1 * A, 0], [0, 1 * A]]
    """
    assert len(arr.shape) == 2 and arr.shape[0] == arr.shape[1]
    N = arr.shape[0]
    tiled = np.zeros((N**2, N**2))
    for i in xrange(N):
        for j in xrange(N):
            if i == j:
                idx_start = N * i
                idx_end = N * (i+1)
                tiled[idx_start:idx_end, idx_start:idx_end] = arr
    return tiled


def permutation_from_transpose(N):
    """
    Construct permutation matrix P such that vec(arr) = P * vec(arr^T) (for any NxN arr)
    See: https://en.wikipedia.org/wiki/Commutation_matrix
    Idea: "where does J_ij vector component get sent?"
    """
    P = np.zeros((N**2, N**2))
    for row in xrange(N):
        for col in xrange(N):
            vec_idx = row * N + col
            vec_transpose_idx = col * N + row
            P[vec_idx, vec_transpose_idx] = 1
    return P


def build_linear_problem(C, D, order='C'):
    """
    Construct the 2D array A and 1D array b in Ax=b
    Based on fluctuation-dissipation relation JC +(JC)^T = -D
    TODO there are many redundant equations, may be better to use the half-vector (upper triangular) vectorize
    Order arg for vectorization style:
        'C' - C-style, do row-by-row (default)
        'F' - Fortran-style, do column-by-column
    """
    # shape checks
    assert C.shape == D.shape and len(C.shape) == 2 and C.shape[0] == C.shape[1] and order in ['C', 'F']
    # symmetry checks
    assert check_symmetric(C) and check_symmetric(D)
    # prep array computation
    N = C.shape[0]
    P = permutation_from_transpose(N)
    # make b vector (RHS)
    b = vectorize_matrix(-D)
    # make A matrix (LHS)
    CxI = arr_cross_eye(C)
    IxC = eye_cross_arr(C)
    if order == 'C':
        A = IxC + np.dot(CxI, P)  # use this is vec(M) is defined row-by-row, i.e. 'C'
    else:
        A = CxI + np.dot(IxC, P)  # use this is vec(M) is defined col-by-col, i.e. 'F'
    return A, b


def solve_regularized_linear_problem(A, b, alpha=0.1, tol=0.0001, verbose=True, use_ridge=False):
    """
    Finds x which minimizes: 1/(2n)*||Ax-b||^2 + alpha*|x| where n is size of b ("number of samples" says scikit)
        - || . || denotes L2-norm
        -  | . |  denotes L1-norm
        - alpha acts as a lagrange multiplier: larger alpha means prioritize smaller J_ij values over ||Ax-b|| error
    Uses scikit-learn Lasso regression: https://scikit-learn.org/stable/modules/linear_model.html
    Default scikit: alpha=0.1, tol=0.0001
    """
    if use_ridge:
        rgr = Ridge(alpha=alpha, tol=tol)
    else:
        rgr = Lasso(alpha=alpha, tol=tol)
    rgr.fit(A, b)
    if verbose:
        if not use_ridge:
            print "rgr.l1_ratio =", rgr.l1_ratio
        print "rgr.tol =", rgr.tol
        print "rgr.coef_\n", rgr.coef_
    return rgr.coef_


def scan_hyperparameter_plot_error(C, D, alpha_low=1e-3, alpha_high=1.0, num=20, check_eig=True):

    def error_fn(J_sol):
        term = np.dot(J_sol, C)
        error_matrix = term + term.T + D
        return np.linalg.norm(error_matrix)  # this is frobenius norm

    A, b = build_linear_problem(C, D)
    alphas = np.linspace(alpha_low, alpha_high, num)
    errors = np.zeros(alphas.shape)
    for idx, alpha in enumerate(alphas):
        x = solve_regularized_linear_problem(A, b, alpha=alpha, tol=0.00001, verbose=False, use_ridge=False)
        J = matrixify_vector(x)
        if check_eig:
            E, V = np.linalg.eig(J)
            print idx, alpha, "eigenvalues", E
        errors[idx] = error_fn(J)
    # plotting
    plt.plot(alphas, errors)
    plt.title('error vs alpha (lagrange multiplier')
    plt.xlabel('alpha')
    plt.xlabel('error (frobenius norm of covariance equation)')
    plt.show()
    return errors


if __name__ == '__main__':
    test_arr_2D = np.array([[1.0, 2.0], [3.0, 4.0]])
    test_arr_3D = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    print '\ntesting inference.py functions...'
    test_arr = test_arr_3D
    N = test_arr.shape[0]
    print 'N =', N
    print 'test_arr\n', test_arr

    print '\ntesting array reshapes on arr...'
    vec_row = vectorize_matrix(test_arr, order='C')
    vec_col = vectorize_matrix(test_arr, order='F')
    print 'vec row-by-row\n', vec_row
    print 'vec col-by-col\n', vec_col
    print 'array row-by-row\n', matrixify_vector(vec_row, order='C')
    print 'array col-by-col\n', matrixify_vector(vec_col, order='F')

    print '\ntesting kronecker products...'
    AxI = arr_cross_eye(test_arr)
    IxA = eye_cross_arr(test_arr)
    print 'arr_cross_eye\n', AxI
    print 'eye_cross_arr\n', IxA

    print '\ntesting permutation...'
    P = permutation_from_transpose(N)
    print 'transposing permutation\n', P
    vec_arr_T = vectorize_matrix(test_arr.T)
    print 'P * vec_arr_T\n', np.dot(P, vec_arr_T)
    print 'vec_arr\n', vec_row

    print '\ntesting Ax=b construction...'
    test_C_arr_2d = np.array([[10.0, 4.0], [4.0, 7.0]])
    test_D_arr_2d = np.array([[2.0, 1.0], [1.0, 2.0]])
    test_C_arr_3d = np.array([[10.0, 4.0, 1.0], [4.0, 7.0, 2.0], [1.0, 2.0, 3.0]])
    test_D_arr_3d = np.array([[2.0, 1.0, 0.5], [1.0, 6.0, 2.0], [0.5, 2.0, 4.0]])
    C = test_C_arr_3d
    D = test_D_arr_3d
    print 'test_C_arr\n', C
    print 'test_D_arr\n', D
    A, b = build_linear_problem(C, D)
    print "A\n", A
    print "b\n", b

    alpha = 0.026
    print '\ntesting Ax=b solution for alpha=%.2e....' % alpha
    x_est = solve_regularized_linear_problem(A, b, alpha=alpha)
    J_est = matrixify_vector(x_est)
    print "x*\n", x_est
    print "J*\n", J_est
    print "check J* solves generating equation JC + (JC)^T + D = 0"
    print "JC + (JC)^T + D\n", np.dot(J_est, C) + np.dot(J_est, C).T + D

    print "\nScanning alphas..."
    errors = scan_hyperparameter_plot_error(C, D, alpha_low=1e-3, alpha_high=0.1, num=2000)
