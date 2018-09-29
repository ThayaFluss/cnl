import numpy as np
import scipy as sp
import os

import sys

from matrix_util import *
from random_matrices import *
import matplotlib.pyplot as plt
from timer import Timer
import time
import logging
from tqdm import tqdm, trange

from cw import *
import copy

from optimizers.adam import Adam
from optimizers.momentum import Momentum
from utils.schedulers import *
from utils.samplers import *
from utils.monitors import *

TEST_C2 =   False

# In[]:
if TEST_C2:
    import spn_c2
    from spn import *
else:
    from spn_c2 import *


# In[]:
i_dpi = 120  #Resolution of figures
i_ext = "png"


def train_fde_spn(dim, p_dim, sample,\
 base_scale = 1e-1, base_lr = 1e-4,\
 max_epoch=400,num_cauchy_rv=1, minibatch_size=1, \
 normalize_sample = False, edge= -1,\
 lr_policy="fix", step_epochs=[],\
 monitor_validation=True,\
 SUBO=True,reg_coef = 0,\
 test_diag_A=-1, test_sigma=-1, \
 monitor_Z= False,test_U= -1, test_V= -1, \
 list_zero_thres=[1e-5,1e-4,1e-3,1e-2,1e-1]):

    if monitor_Z:
        assert (len(sample) == dim )

    sample_size =len(sample)
    assert sample_size == sample.shape[0]
    iter_per_epoch = int(sample_size/minibatch_size)
    max_iter = max_epoch*iter_per_epoch

    if np.allclose(test_diag_A, -1):
        monitor_validation=False

    REG_TYPE = "L1"
    optimizer = "Adam"
    #optimizer = "momentum"
    #lr_policy = "inv"

    if optimizer == "momentum":
        Optimizer = Momentum(lr=base_lr,momentum=0)

    elif optimizer == "Adam":
        Optimizer = Adam(alpha=base_lr)
    else:
        raise ValueError("optimizer is momentum or Adam")



    if lr_policy == "inv":
        Scheduler = Inv()

    elif lr_policy == "step":
        Scheduler = Step(decay=0.1, stepsize=step_epochs[0])
    elif lr_policy == "fix":
        Scheduler = Fix()
    else:
        raise ValueError("lr_policy is inv, step or fix")






    ### tf_summary, logging and  plot
    log_step = 50
    stdout_step = 5000
    plot_stepsize = -1



    ### inputs to be denoised
    if normalize_sample:
        max_row_sample = max(sample)
        normalize_ratio = sp.sqrt(max_row_sample)
        sample /= max_row_sample
    else:
        normalize_ratio = 1.

    sq_sample = sp.sqrt(sample)
    max_sq_sample = max(sq_sample)
    sample_var = np.var(sq_sample)

    ### Cutting off too large parameter
    if edge < 0:
        edge = max_sq_sample*1.01
    ### clip large grad
    clip_grad = -1

    lam_plus = 1 + sp.sqrt(p_dim/dim)
    sigma = max_sq_sample/lam_plus
    sigma = abs(sigma)
    sigma = np.array(sigma)

    diag_A = sq_sample
    diag_A = abs(diag_A)
    diag_A = np.sort(diag_A)
    diag_A = np.array(diag_A, dtype=np.float64)



    logging.info("base_scale = {}".format(base_scale) )
    logging.info("num_cauchy_rv = {}".format(num_cauchy_rv) )
    logging.info("base_lr = {}".format(base_lr) )
    logging.info("minibatch_size = {}".format(minibatch_size) )

    logging.debug("Initial diag_A=\n{}".format(diag_A))
    logging.info("Initial sigma={}".format(sigma))


    ### for monitoring
    if monitor_validation:
        n_test_diag_A = test_diag_A/normalize_ratio
        n_test_sigma = test_sigma/normalize_ratio
        sc_for_plot = SemiCircular(dim=dim,p_dim=p_dim, scale=base_scale)
        sc_for_plot.set_params(n_test_diag_A, n_test_sigma)

    else:
        n_test_diag_A=None
        n_test_sigma=None



    sc = SemiCircular(dim=dim,p_dim=p_dim, scale=base_scale)
    sc.set_params(diag_A, sigma)

    Optimizer_sigma = copy.deepcopy(Optimizer)

    Optimizer.setup(diag_A, Scheduler)
    Optimizer_sigma.setup(sigma, Scheduler)


    if TEST_C2:
        sc2 = spn_c2.SemiCircular(dim=dim,p_dim=p_dim, scale=base_scale)
        sc2.set_params(diag_A, sigma)


    if plot_stepsize > 0:
        x_axis = np.linspace(-max(sample)*4, max(sample)*4, 201) ## Modify
        logging.info("Plotting initial density...")
        if monitor_validation:
            y_axis_truth = sc_for_plot.density_subordinaiton(x_axis)
            sample_for_plot =  sc_for_plot.ESD(num_shot=1,num_cauchy_rv=200)
            plt.hist(sample_for_plot, range=(min(x_axis), max(x_axis)), bins=100, normed=True, label="sampling from a true model \n perturbed by Cauchy($0,\gamma$)",color="pink")
            plt.plot(x_axis,y_axis_truth, linestyle="--", label="true $\gamma$-slice", color="red")


        sc.set_params(diag_A, sigma)
        y_axis_init = sc.density_subordinaiton(x_axis)
        plt.plot(x_axis,y_axis_init, label="Init")
        plt.legend()
        dirname = "../images/train_rnn_sc"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        plt.savefig("{}/plot_density_init.{}".format(dirname,i_ext),dpi=i_dpi)
        plt.clf()
        plt.close()
        logging.info("Plotting initial density...done.")


    scale = base_scale

    ### SGD

    Sampler = CauchyNoiseSampler(scale=scale, num_cauchy_rv=num_cauchy_rv, minibatch_size=minibatch_size)
    Sampler.setup(sample)

    Monitor = SPNMonitor(log_step, stdout_step, plot_stepsize, monitor_validation, \
    monitor_Z, list_zero_thres)
    Monitor.setup(test_diag_A, test_sigma, Optimizer, Sampler, test_U, test_V)


    for n in trange(max_iter):
        update(n ,   sc, SUBO,\
        reg_coef,REG_TYPE,  Optimizer, Optimizer_sigma,Scheduler,Sampler,Monitor,\
        edge)

    logging.info("result:")
    logging.info("sigma:".format(sc.sigma))
    logging.debug("diag_A:\\{}".format(sc.diag_A))
    train_loss_array = np.array(Monitor.train_loss_list)

    if monitor_validation:
        val_loss_array = np.array(Monitor.val_loss_list)
    else:
        val_loss_array = -1

    num_zero_array = np.asarray(Monitor.num_zeros)
    forward_iter = Monitor.total_average_forwad_iter/  (max_iter/log_step)
    ### TODO for sigma
    #list_zero_thres = list_zero_thres[:-1]
    if monitor_validation:
        del sc_for_plot

    diag_A = sc.diag_A
    sigma = sc.sigma
    if normalize_sample:
        sigma *= normalize_ratio
        diag_A *= normalize_ratio

    diag_A = np.sort( abs(diag_A)[::-1])


    result = dict(diag_A = diag_A,\
    sigma= sigma,
    train_loss= train_loss_array,
    val_loss=val_loss_array,
    num_zero=num_zero_array,
    forward_iter= sc.forward_iter)
    del sc

    return result



def update( n , sc, SUBO, \
        reg_coef, REG_TYPE, Optimizer, Optimizer_sigma,Scheduler,Sampler, Monitor,\
        edge):


        ### for epoch base
        # e_idx = int(n/ iter_per_epoch)
        mini = Sampler.get()
        diag_A = sc.diag_A
        sigma = sc.sigma

        ################################
        ### Compute loss and gradients
        ################################
        if SUBO:
            grads, loss = sc.grad_loss_subordination(mini)
            if TEST_C2:
                sc2_grads, sc2_loss = sc2.grad_loss(mini)
                print("")
                print("sub-Psigma:", sc2_grads[-1] - grads[-1])
                print("sc2 psigma:", sc2_grads[-1])
                print("psigma:", grads[-1])
        else :
            grads, loss = sc.grad_loss( mini)

        if reg_coef > 0:
            r_grads, r_loss =  sc.regularization_grad_loss(reg_coef=reg_coef, TYPE=REG_TYPE)
            grads += r_grads
            loss += r_loss


        ##################################
        ### Update Params ###
        ##################################
        Optimizer.update(diag_A, grads[:-1])
        Optimizer_sigma.update(sigma, grads[-1])


        for k in range(diag_A.size):
            if abs(diag_A[k]) > edge:
                logging.info( "diag_A[{}]={} reached the boundary".format(k,diag_A[k]))
                diag_A[k] =edge*0.98
                grads[k] = -1e-8

        if abs(sigma) > edge:
            logging.info("sigma reached the boundary:{}".format(sigma))
            sigma  = edge*0.98
            grads[-1] = -1e-8




        Monitor.collect(loss, sc)
