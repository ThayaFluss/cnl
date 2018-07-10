import numpy as np


from fde_sc_c2 import *
from argparse import ArgumentParser
from train_fde import *


def psvd_cnl(sample_mat, reg_coef=0):
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

    result =  train_fde_sc(dim, p_dim, sample, max_epoch=max_epoch, edge=1.01, reg_coef=reg_coef)

    diag_A = result["diag_A"]
    sigma = result["sigma"]


    out_D = norm*np.sort(diag_A)[::-1]
    out_sigma = sigma*norm
    return U, out_D, V




def rank_estimation(sample_mat,\
 base_scale = 1e-1, dim_cauchy_vec=1, base_lr = 0.1,minibatch_size=1,\
 max_epoch=400, normalize_sample = False,\
 reg_coef = 1e-3):

     p_dim = sample_mat.shape[0]
     dim = sample_mat.shape[1]
     if p_dim < dim:
         sample_mat = sample_mat.T
         p_dim = sample_mat.shape[0]
         dim = sample_mat.shape[1]

     max_epoch*=(p_dim/dim) * (50/dim)
     max_epoch=int(max_epoch)

     _, D, _ = np.linalg.svd(sample_mat)
     sample = D**2 ### eigenvalues of sample_mat.H @ sample_mat

     result = train_fde_sc(
     p_dim = p_dim, dim=dim, sample = sample,\
      base_scale = base_scale, dim_cauchy_vec=dim_cauchy_vec, base_lr = base_lr,minibatch_size=minibatch_size,\
      max_epoch=max_epoch, normalize_sample = normalize_sample,\
      reg_coef = reg_coef,\
      list_zero_thres= [reg_coef] )

     diag_A = result["diag_A"]
     sigma = result["sigma"]
     num_zero_array = result["num_zero"]

     logging.info("list_zero_thres= {}".format( list_zero_thres))
     estimaed_ranks = dim - num_zero_array[-1]

     return estimaed_ranks
