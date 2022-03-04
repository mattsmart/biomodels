import numpy as np


def f_of_x_single_cell(t_scalar, init_cond, single_cell):
    # Gene regulatory dynamics internal to one cell based on its state variables (dx/dt = f(x))
    dxdt = single_cell.ode_system_vector(init_cond, t_scalar)
    return dxdt


def graph_ode_system_vectorized(t_scalar, xvec, single_cell, cellgraph):
    # print("graph_ode_system INPUT LINE SHAPE", xvec.shape)
    xvec_matrix = cellgraph.state_to_rectangle(xvec)
    # Term 1: stores the single cell gene regulation (for each cell)
    #         [f(x_1) f(x_2) ... f(x_M)] as a stacked NM long 1D array
    batch_sz = xvec.shape[-1]  # for vectorized mode of solve_ivp
    term_1 = np.zeros((cellgraph.graph_dim_ode, batch_sz))
    # print("graph_ode_system batch_sz", type(batch_sz), batch_sz)
    # print("graph_ode_system t_scalar", type(t_scalar), t_scalar)
    # print("graph_ode_system xvec", type(xvec), xvec.shape)
    # print("graph_ode_system xvec_matrix", type(xvec_matrix), xvec_matrix.shape)

    for cell_idx in range(cellgraph.num_cells):
        a = cellgraph.sc_dim_ode * cell_idx
        b = cellgraph.sc_dim_ode * (cell_idx + 1)
        xvec_sc = xvec_matrix[:, cell_idx]
        # print(xvec_sc.shape)
        term_1[a:b, :] = f_of_x_single_cell(t_scalar, xvec_sc, single_cell)

    # TODO check that slicing is correct with vectorized batching
    # TODO this can be parallelized as one linear Dvec * np.dot(X, L^T) -- see graph_ode_system()
    # Term 2: stores the cell-cell coupling which is just laplacian diffusion -c * L * x
    # Note: we consider each reactant separately with own diffusion rate
    term_2 = np.zeros((cellgraph.graph_dim_ode, batch_sz))
    for gene_idx in range(cellgraph.sc_dim_ode):
        indices_for_specific_gene = np.arange(gene_idx, cellgraph.graph_dim_ode, cellgraph.sc_dim_ode)
        xvec_specific_gene = xvec[indices_for_specific_gene]
        diffusion_specific_gene = - cellgraph.diffusion[gene_idx] * np.dot(cellgraph.laplacian, xvec_specific_gene)
        term_2[indices_for_specific_gene, :] = diffusion_specific_gene

    dxvec_dt = term_1 + term_2
    # print("graph_ode_system OUTPUT LINE SHAPE", dxvec_dt.shape)
    return dxvec_dt


def graph_ode_system(t_scalar, xvec, single_cell, cellgraph):
    """
    Non-vectorized implementation of graph_ode_system_vectorized()
    """
    xvec_matrix = cellgraph.state_to_rectangle(xvec)
    # Term 1: stores the single cell gene regulation (for each cell)
    #         [f(x_1) f(x_2) ... f(x_M)] as a stacked NM long 1D array
    term_1 = np.zeros(cellgraph.graph_dim_ode)
    # TODO can this be sped up?
    for cell_idx in range(cellgraph.num_cells):
        a = cellgraph.sc_dim_ode * cell_idx
        b = cellgraph.sc_dim_ode * (cell_idx + 1)
        xvec_sc = xvec_matrix[:, cell_idx]
        term_1[a:b] = f_of_x_single_cell(t_scalar, xvec_sc, single_cell)

    # Term 2: stores the cell-cell coupling which is just laplacian diffusion -c * L * x
    # Note: we consider each reactant separately with own diffusion rate
    X_times_LT = np.matmul(xvec_matrix, cellgraph.laplacian.T)
    # Note: the following line is equivalent to -np.matmul(cellgraph.diffusion_diag_matrix, X_times_LT)
    # - multiplying by a diagonal matrix can be spedup via broadcasting to multiply by constant rows
    D_times_X_times_LT = - cellgraph.diffusion[:, None] * X_times_LT
    term_2 = cellgraph.state_to_stacked(D_times_X_times_LT)
    """ Old, slower way
    term_2 = np.zeros(cellgraph.graph_dim_ode)
    for gene_idx in range(cellgraph.sc_dim_ode):
        indices_for_specific_gene = np.arange(gene_idx, cellgraph.graph_dim_ode, cellgraph.sc_dim_ode)
        xvec_specific_gene = xvec[indices_for_specific_gene]
        diffusion_specific_gene = - cellgraph.diffusion[gene_idx] * np.dot(cellgraph.laplacian, xvec_specific_gene)
        term_2[indices_for_specific_gene] = diffusion_specific_gene"""

    dxvec_dt = term_1 + term_2
    return dxvec_dt
