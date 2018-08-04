import numpy as np


from spn_c2 import *
#from argparse import ArgumentParser
from train_fde import *


def psvd_cnl(sample_mat, reg_coef=0, minibatch_size=1):
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

    result =  train_fde_spn(dim, p_dim, sample, max_epoch=max_epoch, edge=1.01, reg_coef=reg_coef,\
    dim_cauchy_vec=minibatch_size)

    diag_A = result["diag_A"]
    sigma = result["sigma"]


    out_D = norm*np.sort(diag_A)[::-1]
    out_sigma = sigma*norm
    return U, out_D, V, out_sigma



def rank_estimation(sample_mat,reg_coef=1e-3, minibatch_size=1):

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

    list_zero_thres = [reg_coef]
    result =  train_fde_spn(dim, p_dim, sample, max_epoch=max_epoch, edge=1.01, reg_coef=reg_coef,\
    dim_cauchy_vec=minibatch_size, list_zero_thres=list_zero_thres)

    diag_A = result["diag_A"]
    sigma = result["sigma"]
    num_zero_array = result["num_zero"]

    logging.info("list_zero_thres= {}".format( list_zero_thres))
    estimaed_ranks = dim - num_zero_array[-1]

    return estimaed_ranks, diag_A, sigma



def z_value(sample_mat, a, s):
    assert s != 0
    U, singular, V = np.linalg.svd(sample_mat)
    p = sample_mat.shape[0]
    d = sample_mat.shape[1]
    Diff = U @ rectangular_diag(singular - a, p, d) @ V
    z = np.sum(Diff)/ (sp.sqrt(p)*s)
    return z
