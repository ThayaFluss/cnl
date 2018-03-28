import numpy as np
import scipy as sp

from matrix_util import *
from random_matrices import *
import matplotlib.pyplot as plt
from timer import Timer
import os
import time
import logging

"""
#to console
stream_log = logging.StreamHandler()
stream_log.setLevel(logging.INFO)

stream_log.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))


#to file
file_log = logging.FileHandler('log/test_loger.txt', 'w')
file_log.setLevel(logging.INFO)
file_log.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
# root logger
logging.getLogger().addHandler(stream_log)
logging.getLogger().addHandler(file_log)
# level of root logger must minumum of loggers
logging.getLogger().setLevel(logging.INFO)
"""

from cauchy import *


def test_optimize(size, num_sample,  max_epoch,base_lr,minibatch,eps,\
jobname="test_optimize",random_minibatch=False, online_sampling=False, ):
    #to file
    file_log = logging.FileHandler('log/{}.log'.format(jobname), 'w')
    file_log.setLevel(logging.INFO)
    file_log.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(file_log)
    #logging.getLogger().setLevel(logging.INFO)
    #param = 2 + np.random.rand(size)*3
    #param = np.sort(param)
    #"""
    param = np.ones(size)*2
    #param = 2*np.random_sample(size)
    half_size = int(size/2)
    for i in range(half_size):
        param[i] = 5
    #"""
    logging.info( "truth=\n{}".format(np.array(param)))
    param = np.sort(param)
    param_mat = np.diag(param)
    sigma = 1
    evs_list =[]
    for i  in range(num_sample):
        evs= np.linalg.eigh(info_plus_noise(size, param_mat,sigma, COMPLEX=True))[0]
        evs_list += evs.tolist()
    if num_sample == 1:
        logging.info( "sqrt(sample)=\n{}".format(sp.sqrt(np.array(evs_list))))

    #mean_param = 2
    sq_sample = np.array(sp.sqrt(evs_list))
    mean_param =np.average(sq_sample)

    sc = SemiCircular(update_sigma=False, minibatch=minibatch, eps=eps)

    result, r_v= sc.optimize(size, evs_list, init_sigma=sigma, init_mean_diag_A = mean_param,max_epoch=max_epoch,base_lr=base_lr, test_diag_A=param, test_sigma=sigma, random_minibatch=random_minibatch, online_sampling=online_sampling)
    plt.figure()

    plt.plot(param, label="Truth")
    plt.plot(sq_sample, label="Sample")
    plt.plot(result, label="Result")
    plt.legend()
    plt.savefig("images/test_optimize_lr_1e-2_scale_1e-2/{}.png".format(jobname))
    logging.getLogger().removeHandler(file_log)

test_optimize(size=36, num_sample=1, max_epoch=500, base_lr=0.01, \
minibatch=4, eps=1e-2)

#for random_minibatch in [True, False]:
#    for minibatch in [8,16,32]:
#        jobname = "5-2_32x32_ss_mb-{}_rm-{}".format(minibatch, random_minibatch)
#        test_optimize(size=32, num_sample=1, minibatch=minibatch, random_minibatch=random_minibatch, jobname=jobname)
