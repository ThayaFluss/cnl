import numpy as np


from fde_sc_c2 import *
from argparse import ArgumentParser
from train_fde import *

def prob_svd_cnl(sample_mat):
    p_dim = sample_mat.shape[0]
    dim = sample_mat.shape[1]
    if p_dim < dim:
        sample_mat = sample_mat.T
        p_dim = sample_mat.shape[0]
        dim = sample_mat.shape[1]

    U, D, V = np.linalg.svd(sample_mat)
    norm = max(D)
    D /= norm

    sample = D**2
    max_epoch = 8*p_dim
    reg_coef = 0

    diag_A, sigma, _, _, _, _ \
    =  train_fde_sc(dim, p_dim, sample, max_epoch=max_epoch, edge=1.01, reg_coef=reg_coef)

    out_D = norm*np.sort(diag_A)[::-1]
    out_sigma = sigma*norm
    import pdb; pdb.set_trace()

    return U, out_D, V
