import numpy as np
import scipy as sp

from matrix_util import *
from random_matrices import *
import matplotlib.pyplot as plt
from timer import Timer
import os
import time
import logging
import sys
import json
from tqdm import tqdm,trange
from datetime import datetime
from train_spn import *



"""
True params : scalar*rectangular_diag(1) , sigma
params:  in the same form same form
"""

def main():

    p = 100
    d = 100

    true_as = [ 0.2,0.4, 0.6,  0.8,  1.0]
    true_sigmas = [0.1]

    true = []
    result_L2 = []
    result_CNL= []
    for a in true_as:
        for sigma in true_sigmas:
            true.append([a,sigma])
            compare(p,d,a,sigma, result_L2, result_CNL)

    true = np.asarray(true)
    result_L2 = np.asarray(result_L2)
    result_CNL = np.asarray(result_CNL)


    logging.debug("\n{}".format(result_L2 -true))
    logging.debug("\n{}".format(result_CNL - true))

    dirname = "../images/vtss"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    """
    a-s scatter plot
    """
    plt.figure()
    plt.xlabel("a")
    plt.ylabel("s")
    plt.scatter(true[:,0], true[:,1], label="T")
    plt.scatter(result_L2[:,0], result_L2[:,1], label="L2")
    plt.scatter(result_CNL[:,0], result_CNL[:, 1], label="CNL")
    plt.legend()

    plt.savefig("{}/L2vsCNL.png".format(dirname))


    """
    a plot
    """
    plt.figure()
    plt.ylabel("a")
    plt.plot(true[:,0],label="T")
    plt.plot(result_L2[:,0], label="L2")
    plt.plot(result_CNL[:, 0], label="CNL")
    plt.legend()
    plt.savefig("{}/a.png".format(dirname))


def compare(p,d,a, sigma, result_L2, result_CNL):

    param = a*np.ones(d)  ## true parameter
    param_mat = rectangular_diag(param, p_dim=p, dim=d)
    sample_mat = signal_plus_noise(param_mat, sigma, RAW=True)

    a_L2, s_L2 = estimation_L2(sample_mat)
    a_CNL, s_CNL = estimation_CNL(sample_mat, param, sigma)

    logging.info( "a_L2:{} vs a_CNL:{}".format(a_L2, a_CNL))
    logging.info( "s_L2:{} vs s_CNL:{}".format(s_L2, s_CNL))

    result_L2.append( [a_L2, s_L2])
    result_CNL.append( [a_CNL, s_CNL])


def estimation_L2(sample_mat):
    p = sample_mat.shape[0]
    d = sample_mat.shape[1]

    temp = 0
    for i in range(d):
        temp += sample_mat[i,i]

    r_a = temp/d

    sample_var =  0
    for i in range(p):
        for j in range(d):
            if i == j and i < d:
                sample_var += (sample_mat[i,i] -  r_a)**2
            else:
                sample_var += sample_mat[i,j]**2

    r_v = d*sample_var/(d*p)
    r_s = sp.sqrt(r_v)

    return r_a,r_s

"""
@param : used only for  monitoring
"""
def estimation_CNL(sample_mat, param, sigma):
    p = sample_mat.shape[0]
    d = sample_mat.shape[1]
    U, evs, V = np.linalg.svd(sample_mat)
    sample = np.asarray(evs)

    kwargs = {
        "base_scale":           1e-1,
        "num_cauchy_rv":        1,
        "base_lr":              1e-4,
        "minibatch_size":       1,
        "max_epoch":            200,
        "reg_coef":             0,
        "optimizer":            "Adam",
        "lr_policy":            "inv",
        "step_epochs":           [150],
        "monitor_validation":   True,
        "test_diag_A": param,
        "test_sigma": sigma,
        "test_U": U,
        "test_V": V,
        "monitor_Z":True,
        #list_zero_thres = None,\
        "scalar_initialize":  True}

    result= train_fde_spn(d, p,  sample, **kwargs)

    r_diag_A        =result["diag_A"]
    r_sigma         =result["sigma"]
    #train_loss_array=result["train_loss"]
    val_loss_array  =result["val_loss"]
    plt.plot(val_loss_array)
    plt.savefig("../images/vtss/val_loss.png")
    #import pdb; pdb.set_trace()
    #num_zero_array  =result["num_zero"]
    #forward_iter    =result["forward_iter"]

    logging.info("estimated a: {}".format(r_diag_A[0]) )
    logging.info("estimated sigma: {}".format(r_sigma))

    return r_diag_A[0], r_sigma



if __name__ == "__main__": main()
