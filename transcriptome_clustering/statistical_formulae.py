import numpy as np


def build_diffusion(params):
    p = params
    D = np.zeros((p.dim, p.dim))
    # TODO
    return D


def build_covariance(params):
    p = params
    cov = np.zeros((p.dim, p.dim))
    # TODO
    return cov


def infer_interactions(params):
    p = params
    J = np.zeros((p.dim, p.dim))
    # TODO
    return J
