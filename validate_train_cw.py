import numpy as np
import scipy as sp

from matrix_util import *
from random_matrices import *
import matplotlib.pyplot as plt
from timer import Timer
import os
import time
import logging

from argparse import ArgumentParser
from train_fde import *


import env_logger

def options(logger=None):
    desc   = u'{0} [Args] [Options]\nDetailed options -h or --help'.format(__file__)
    parser = ArgumentParser(description = desc)

    # options
    parser.add_argument('-d', '--dim',
                        type     = int,
                        dest     = 'dim',
                        required = False,
                        default  =  50,
                        help     = "column")
    parser.add_argument('-p', '--p_dim',
                        type     = int,
                        dest     = 'p_dim',
                        required = False,
                        default  =  50,
                        help     = "row")
    parser.add_argument('-m', '--minibatch',
                        type     = int,
                        dest     = 'minibatch',
                        required = False,
                        default  =  1,
                        help     = "minibatch_size")
    parser.add_argument('-j', '--jobname',
                        type     = str,
                        dest     = 'jobname',
                        required = False,
                        default  =  50,
                        help     = "min_singular")
    parser.add_argument('-nt', '--num_test',
                        type     = int,
                        dest     = 'num_test',
                        required = False,
                        default  =  10,
                        help     = "Number of tests")
    parser.add_argument('-me', '--max_epoch',
                        type     = int,
                        dest     = 'max_epoch',
                        required = False,
                        default  =  120,
                        help     = "max_epoch")
    parser.add_argument('-dpi', '--dpi',
                        type     = int,
                        dest     = 'dpi',
                        required = False,
                        default  =  600,
                        help     = "Resolution of figures (default:60)")

    return parser.parse_args()

opt =options()


i_dim = min(opt.dim, opt.p_dim)
i_p_dim = max(opt.dim, opt.p_dim)
jobname = opt.jobname
i_dpi = opt.dpi






def test_optimize(\
    base_scale ,dim_cauchy_vec, \
    base_lr=1e-1 ,  minibatch_size=1,  max_epoch=20,\
    jobname="test_optimize_cw",\
    min_singular=0,
    dim=i_dim,
     zero_dim=16,
     COMPLEX = False
    ):
    dirname = "./images/{}".format(jobname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    num_sample =1
    #to file
    file_log = logging.FileHandler('log/{}.log'.format(jobname), 'w')
    file_log.setLevel(logging.INFO)
    file_log.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(file_log)
    #logging.getLogger().setLevel(logging.INFO)
    #param = 2 + np.random.rand(size)*3
    #param = np.sort(param)
    #"""
    p_dim = dim
    ###marhcnko 1 + \sqrt{p/d}
    MP_ratio = p_dim/dim
    high_singular = 1./ ( 1 + sp.sqrt(MP_ratio))
    assert min_singular < high_singular
    param = np.random.uniform(low=min_singular, high=high_singular, size = p_dim)
    #param = 2*np.random_sample(dim)
    for i in range(zero_dim):
        param[i] = 0.
    #"""
    param = np.sort(param)
    logging.info( "zero_dim={}".format(zero_dim) )
    logging.info( "min_singular={}".format(min_singular) )
    logging.info( "truth=\n{}".format(np.array(param)))

    ### For generate sample, we do not need scale.
    cw_for_sample = CompoundWishart(dim = dim, p_dim=p_dim, scale = 1e-9)
    cw_for_sample.b = param
    plt.rc("text", usetex=True)

    evs_list = cw_for_sample.ESD(num_shot = num_sample, COMPLEX=COMPLEX)

    if num_sample == 1:
        logging.info( "sample=\n{}".format(np.asarray(evs_list)))




    b, train_loss_array, val_loss_array= train_fde_cw(dim, p_dim,\
        sample=evs_list,\
        base_scale=base_scale ,\
        dim_cauchy_vec=dim_cauchy_vec,\
        base_lr =base_lr ,\
        minibatch_size=minibatch_size,\
        max_epoch=max_epoch,\
        monitor_validation=True,\
        test_b=param)

    plt.figure()
    plt.plot(np.sort(param), label="Truth")
    plt.plot(np.sort(evs_list), label="Sample")
    plt.plot(np.sort(b), label="Result")
    plt.legend()

    dirname = "images/val_train_cw"
    if not os.path.exists(dirname):
            os.makedirs(dirname)

    plt.savefig("{}/{}.png".format(dirname, jobname),dpi=i_dpi)
    plt.clf()
    logging.getLogger().removeHandler(file_log)

    epoch = int(i_dim/minibatch_size)
    train_loss_array = train_loss_array.reshape([-1,epoch]).mean(axis=1)
    val_loss_array = val_loss_array.reshape([-1,epoch]).mean(axis=1)

    return b, train_loss_array, val_loss_array


def test_scale_balance():
    num_test = opt.num_test
    ### MP-ratio
    min_singular = - 1./( 1  + np.sqrt(1))
    zero_dim = 0
    max_epoch = opt.max_epoch
    base_lr = 1e-2
    minibatch_size = opt.minibatch
    ### TODO for paper
    list_base_scale =[ 1e-1/4, 1e-1/2, 1e-1, 2e-1, 4e-1]
    #list_base_scale =[ 1e-1]
    list_dim_cauchy_vec =  [1]
    ### for test
    #list_base_scale =[ 1e-1]
    #list_dim_cauchy_vec =  [1024]

    list_val_loss_array = []
    list_train_loss_array = []
    for base_scale in list_base_scale:
        for dim_cauchy_vec in list_dim_cauchy_vec:
            average_b = 0;  average_val_loss=0;average_train_loss=0;
            for n in range(num_test):
                b, train_loss_array, val_loss_array=test_optimize(\
                base_lr = base_lr,minibatch_size=minibatch_size,\
                 max_epoch=max_epoch,\
                min_singular=min_singular, zero_dim = zero_dim, \
                base_scale=base_scale, dim_cauchy_vec=dim_cauchy_vec)
                average_b += b
                average_val_loss += val_loss_array
                average_train_loss += train_loss_array
            average_b /= num_test
            average_val_loss /= num_test
            average_train_loss /= num_test
            logging.info("RESULT:base_scale = {}, ncn = {}, val_loss = {},\n  average_b = \n{}".format(\
            base_scale, dim_cauchy_vec, average_val_loss[-1], average_b) )
            list_val_loss_array.append(average_val_loss)
            list_train_loss_array.append(average_train_loss)

    plt.figure()
    plt.rc('font', family='serif', serif='Times')
    plt.rc('text', usetex=True)
    plt.rcParams["font.size"] = 8*2

    if len(list_dim_cauchy_vec) == 1:
        plt.title("$Noise dimension = {}$".format(list_dim_cauchy_vec[0]))
    x_axis = np.arange(average_val_loss.shape[0])

    n = 0
    for base_scale in list_base_scale:
        for dim_cauchy_vec in list_dim_cauchy_vec:
            plt.plot(x_axis,list_val_loss_array[n], label="({}, {})".format(base_scale, dim_cauchy_vec))
            n+=1
    #plt.title("Validation loss")
    plt.xlabel("epoch")
    plt.ylabel("validation loss")
    plt.ylim(0.2, 1.2)
    plt.legend()
    dirname = "./images/scale_balance_cw"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = "{}/test_val.png".format(dirname)
    logging.info(filename)
    plt.savefig(filename,dpi=i_dpi)
    plt.clf()
    plt.close()

    n = 0
    for base_scale in list_base_scale:
        for dim_cauchy_vec in list_dim_cauchy_vec:
            plt.plot(x_axis,list_train_loss_array[n], label="({}, {})".format(base_scale, dim_cauchy_vec))
            n+=1
    #plt.title("Validation loss")
    plt.xlabel("epoch")
    plt.ylabel("train loss")
    plt.legend()
    filename = "{}/test_train.png".format(dirname)
    plt.savefig(filename,dpi=i_dpi)
    plt.clf()
    plt.close()



timer = Timer()
timer.tic()
test_scale_balance()

timer.toc()
logging.info("total_time:{}".format(timer.total_time))
