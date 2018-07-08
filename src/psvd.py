import numpy as np


from fde_sc_c2 import *
from argparse import ArgumentParser
from train_fde import *


def psvd_cnl(sample_mat):
    """ probabilistic singular value decomposition """
    """ by Cauchy Noise Loss """ 
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

    max_epoch = (20000/dim)  * (p_dim/dim)
    max_epoch = int(max_epoch)
    reg_coef = 1e-3

    diag_A, sigma, _, _, _, _ \
    =  train_fde_sc(dim, p_dim, sample, max_epoch=max_epoch, edge=1.01, reg_coef=reg_coef)

    out_D = norm*np.sort(diag_A)[::-1]
    out_sigma = sigma*norm
    return U, out_D, V
