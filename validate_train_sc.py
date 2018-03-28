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
from tqdm import tqdm,trange
from datetime import datetime
from train_fde import *
from vbmf.vbmf import VBMF2
from vbmf.validate_vbmf_ipn import validate_vbmf_ipn


from argparse import ArgumentParser
import logging

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
    parser.add_argument('-v', '--vbmf',
                        type     = bool,
                        dest     = 'vbmf',
                        required = False,
                        default  =  False,
                        help     = "empirical vbmf")
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

    return parser.parse_args()

opt =options()


i_dim = min(opt.dim, opt.p_dim)
i_p_dim = max(opt.dim, opt.p_dim)
i_minibatch_size = opt.minibatch
jobname = opt.jobname
i_vbmf = opt.vbmf


if not jobname in ["min_singular" , "scale_balance"]:
    jobname = "min_singular"

def test_optimize(\
    base_scale ,dim_cauchy_vec, \
    base_lr=1e-1 ,  minibatch_size=i_minibatch_size,  max_epoch=20,\
    reg_coef=0,\
    jobname="train_v2",\
    min_singular=0,
    dim=i_dim,p_dim = i_p_dim,\
     zero_dim=16,\
    list_zero_thres=[],
    COMPLEX=False,SUBO=False):
    dirname = "./images/{}".format(jobname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    num_sample =1
    sigma = 0.1

    logging.info( "zero_dim={}".format(zero_dim) )
    logging.info( "min_singular={}".format(min_singular) )

    param_mat = random_from_diag(p_dim, dim, zero_dim, min_singular, COMPLEX)
    _, param, _ = np.linalg.svd(param_mat)
    logging.info( "truth=\n{}".format(np.sort(param)))

    evs_list =[]
    for i  in range(num_sample):
        evs= np.linalg.eigh(info_plus_noise(param_mat,sigma, COMPLEX=False))[0]
        evs_list += evs.tolist()
    if num_sample == 1:
        logging.info( "sqrt(sample)=\n{}".format(sp.sqrt(np.array(evs_list))))
    #mean_param = 2
    sq_sample = np.array(sp.sqrt(evs_list))
    mean_param =np.average(sq_sample)


    sample = np.asarray(evs_list)

    r_diag_A, r_sigma, train_loss_array, val_loss_array,num_zero_array\
    = train_fde_sc(dim, p_dim,  sample,\
        base_scale=base_scale ,\
        dim_cauchy_vec=dim_cauchy_vec,\
        base_lr =base_lr ,\
        minibatch_size=minibatch_size,\
        max_epoch=max_epoch,\
        reg_coef=reg_coef,\
        monitor_validation=True,\
        test_diag_A=param,\
        test_sigma =sigma,\
        list_zero_thres = list_zero_thres,\
        SUBO=SUBO)

    plt.figure()
    plt.plot(param, label="Truth")
    plt.plot(sq_sample, label="Sample")
    plt.plot(r_diag_A, label="Result")
    plt.legend()
    plt.savefig("images/train_v2/{}.png".format(jobname))
    plt.clf()
    #logging.getLogger().removeHandler(file_log)


    epoch = int(dim/minibatch_size)
    train_loss_array = train_loss_array.reshape([-1, epoch]).mean(axis=1)
    val_loss_array = val_loss_array.reshape([-1,epoch]).mean(axis=1)
    num_zero_array = num_zero_array.reshape([-1,epoch, len(list_zero_thres)]).mean(axis=1)

    return r_diag_A, r_sigma, train_loss_array, val_loss_array, num_zero_array

#@param jobname = "min_singular" or "scale_balance"
def test_sc(jobname="min_singular", SUBO=True, VS_VBMF=False):
    num_test = opt.num_test
    max_epoch = opt.max_epoch
    max_epoch= int(max_epoch*i_p_dim/i_dim)
    base_scale = 0.2
    base_scale *= i_p_dim/i_dim
    base_lr = 0.1
    ### for paper
    reg_coef = 2e-4
    #reg_coef = 0
    if jobname == "scale_balance":
        ### set zero_thres for max_epoch=120
        ### TODO for paper
        #list_base_scale =[ 0.5*1e-1,1e-1, 2*1e-1 ]
        list_dim_cauchy_vec =  [1]
        list_base_scale = [ base_scale/2, base_scale, base_scale*2]
        ###fix
        list_min_singular = [0.5]
        list_zero_dim = [int(i_dim/2)]

        list_zero_thres = [1e-4,1e-2]
    elif jobname == "min_singular":
        ### for paper
        list_zero_dim = [10,20,30,40]
        list_min_singular=[0.05,0.1, 0.2,0.3,0.4]
        list_base_scale = [base_scale]
        list_dim_cauchy_vec = [2] ### can be larger than dim.
        list_zero_thres = [1e-4,1e-2]

        ### for debug
        ### scale * lr /dim_cauchy = 0.01/16
        #m = int(i_dim/8)
        #list_zero_dim = [20, 40]
        #list_min_singular=[ 0.2, 0.3]
        #list_base_scale = [0.2*i_p_dim/i_dim]
        #list_dim_cauchy_vec = [2]
    else:
        sys.exit(-1)

    now = datetime.now()

    temp_jobname = jobname + '_{0:%m%d%H%M}'.format(now)
    dirname = "images/{}".format(temp_jobname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    setting_log = open("{}/setting.log".format(dirname), "w")
    setting_log.write("jobname:{}\n".format(temp_jobname))
    setting_log.write("dim:{}\n".format(i_dim))
    setting_log.write("p_dim:{}\n".format(i_p_dim))
    setting_log.write("minibatch:{}\n".format(i_minibatch_size))
    setting_log.write("num_test:{}\n".format(num_test))
    setting_log.write("max_epoch:{}\n".format(max_epoch         ))
    setting_log.write("base_lr:{}\n".format(base_lr           ))
    setting_log.write("reg_coef:{}\n".format(reg_coef          ))
    setting_log.write("list_dim_cauchy_vec:{}\n".format(list_dim_cauchy_vec))
    setting_log.write("list_base_scale:{}\n".format(list_base_scale   ))
    setting_log.write("list_min_singular:{}\n".format(list_min_singular ))
    setting_log.write("list_zero_dim:{}\n".format(list_zero_dim     ))
    setting_log.write("list_zero_thres:{}\n".format(list_zero_thres   ))
    setting_log.close()


    num_zero_dim = len(list_zero_dim)
    num_min_singular = len(list_min_singular)


    def _mean_and_var(results):
        m = np.mean(results, axis = 0)
        v = np.mean( (results -m)**2, axis=0)
        return m,v


    if VS_VBMF:
        ### Run VBMF
        vbmf_estimated_rank = np.empty((num_zero_dim, num_min_singular))
        for i in range(num_zero_dim):
            zero_dim = list_zero_dim[i]
            for j in range(num_min_singular):
                min_singular = list_min_singular[j]
                results = []
                for n in range(num_test):
                    r = validate_vbmf_ipn(i_dim,i_p_dim, 0.1, min_singular, zero_dim)
                    results.append(r)

                m_r, v_r = _mean_and_var(results)
                vbmf_estimated_rank[i][j] =  m_r

        ### plot results
        x_axis = list_min_singular
        for i in range(num_zero_dim):
            plt.figure()
            zero_dim = list_zero_dim[i]
            true_rank = i_dim - zero_dim
            plt.plot(x_axis, vbmf_estimated_rank[i], label="true rank: {}".format(true_rank), marker="+")
            plt.xlabel("min singular")
            plt.ylabel("Estimated rank")
            plt.legend()
            plt.savefig("{}/vbmf_true_ranks-{}.png".format(dirname,true_rank ))
            plt.clf()
            plt.close()




    ### Run
    num_thres = len(list_zero_thres)

    list_val_loss_curve = []
    list_train_loss_curve = []
    list_estimated_rank_curve  = []


    for base_scale in list_base_scale:
        for dim_cauchy_vec in list_dim_cauchy_vec:
            for zero_dim in list_zero_dim:
                for min_singular in list_min_singular:
                    result_diag_A = []; result_sigma=[]; result_val_loss=[];result_train_loss=[];
                    result_num_zero = []
                    for n in trange(num_test, desc="num_test"):
                        diag_A, sigma, train_loss_array, val_loss_array,num_zero_array\
                        =test_optimize(\
                        base_lr=base_lr,
                        max_epoch=max_epoch,min_singular=min_singular, zero_dim = zero_dim,\
                        base_scale=base_scale, dim_cauchy_vec=dim_cauchy_vec,\
                        reg_coef=reg_coef,\
                        list_zero_thres= list_zero_thres,
                        SUBO=SUBO)
                        result_diag_A.append( diag_A)
                        result_sigma.append(sigma)
                        result_val_loss.append( val_loss_array)
                        result_train_loss.append( train_loss_array)
                        #result_num_zero.append( num_zero_array.T[idx_zero_thres_for_plot])
                        result_num_zero.append( num_zero_array)
                    m_diag_A, v_diag_A = _mean_and_var(result_diag_A)
                    m_sigma, v_sigma = _mean_and_var(result_sigma)
                    m_train_loss, v_train_loss = _mean_and_var(result_train_loss)
                    m_val_loss, v_val_loss = _mean_and_var(result_val_loss)
                    m_num_zero, v_num_zero = _mean_and_var(result_num_zero)

                    logging.info("RESULT:base_scale = {}, ncn = {}, val_loss = {},\n average_sigma = {}, average_diag_A = \n{}".format(\
                    base_scale, dim_cauchy_vec, m_val_loss[-1], m_sigma, m_diag_A) )
                    list_train_loss_curve.append(m_train_loss)
                    list_val_loss_curve.append(m_val_loss)
                    list_estimated_rank_curve.append( i_dim - m_num_zero )

    x_axis = np.arange(m_val_loss.shape[0])

    if jobname == "scale_balance":
        #############
        ###plot val loss
        #############
        plt.figure()
        n = 0
        for base_scale in list_base_scale:
            for dim_cauchy_vec in list_dim_cauchy_vec:
                plt.plot(x_axis,list_val_loss_curve[n], label="({}, {})".format(base_scale, dim_cauchy_vec))
                n+=1
        plt.legend()

        plt.title("Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Validation loss")

        plt.savefig("{}/test_scale_val.png".format(dirname))
        plt.clf()
        plt.close()


        #############
        ###plot train loss
        #############
        plt.figure()
        n = 0
        for base_scale in list_base_scale:
            for dim_cauchy_vec in list_dim_cauchy_vec:
                plt.plot(x_axis,list_train_loss_curve[n], label="({}, {})".format(base_scale, dim_cauchy_vec))
                n+=1

        plt.legend()
        plt.title("Train loss")
        plt.xlabel("Epoch")
        plt.ylabel("Train loss")
        plt.savefig("{}/test_scale_train.png".format(dirname))

        plt.clf()
        plt.close()

    elif jobname == "min_singular":
        #import pdb; pdb.set_trace()
        num = len(list_zero_dim)
        half = int(num /2)

        ################
        ###plot val_loss
        ################
        n = 0

        plt.figure()
        for zero_dim in list_zero_dim[:half]:
            for min_singular in list_min_singular:
                plt.plot(x_axis,list_val_loss_curve[n], label="({}, {})".format(i_dim - zero_dim, min_singular))
                n+=1

        plt.title("Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Validation loss")

        plt.legend()
        plt.savefig("{}/test_ms_low_val.png".format(dirname))
        plt.clf()
        plt.close()

        plt.figure()
        for zero_dim in list_zero_dim[half:]:
            for min_singular in list_min_singular:
                plt.plot(x_axis,list_val_loss_curve[n], label="({}, {})".format(i_dim -zero_dim, min_singular))
                n+=1

        plt.title("Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Validation loss")

        plt.legend()
        plt.savefig("{}/test_ms_high_val.png".format(dirname))
        plt.clf()
        plt.close()


        ################
        ###plot estimated rank
        ################
        for k in range(len(list_zero_thres)):
            n = 0
            logging.info(list_estimated_rank_curve)
            plt.figure()
            for zero_dim in list_zero_dim[:half]:
                for min_singular in list_min_singular:
                        plt.plot(x_axis,list_estimated_rank_curve[n][:,k],\
                        label="({}, {})".format(\
                        i_dim - zero_dim, min_singular))
                        n+=1
                plt.plot(x_axis, (i_dim-zero_dim)*np.ones(len(x_axis)), linestyle="--", color="black")

            #plt.title("Validation loss")
            plt.xlabel("Epoch")
            plt.ylabel("Estimated rank")
            plt.ylim(0,i_dim)

            plt.legend()
            name = "{}/test_ms_low_zeros_thres-{}.png".format(dirname, list_zero_thres[k])
            logging.info(name)
            plt.savefig(name)
            plt.clf()
            plt.close()

            plt.figure()
            for zero_dim in list_zero_dim[half:]:
                for min_singular in list_min_singular:
                    plt.plot(x_axis,list_estimated_rank_curve[n][:,k], label="({}, {})".format(i_dim - zero_dim, min_singular))
                    n+=1
                plt.plot(x_axis, (i_dim - zero_dim)*np.ones(len(x_axis)), linestyle="--", color="black")

            #plt.title("Validation loss")
            plt.title("Estimated rank")
            plt.xlabel("Epoch")
            plt.ylabel("Estimated rank")
            plt.ylim(0,i_dim)
            #plt.yticks( np.arange(5, i_dim+1))
            plt.legend()
            name = "{}/test_ms_high_zeros_thres-{}.png".format(dirname, list_zero_thres[k])
            logging.info(name)
            plt.savefig(name)
            plt.clf()
            plt.close()





        ##################
        ###plot train_loss
        ##################
        n = 0

        plt.figure()
        for zero_dim in list_zero_dim[:half]:
            for min_singular in list_min_singular:
                plt.plot(x_axis,list_train_loss_curve[n], label="({}, {})".format(i_dim - zero_dim, min_singular))
                n+=1

        #plt.title("Validation loss")
        plt.xlabel("epoch")
        plt.ylabel("train loss")
        plt.legend()

        plt.savefig("{}/test_ms_low_train.png".format(dirname))
        plt.clf()
        plt.close()

        plt.figure()
        for zero_dim in list_zero_dim[half:]:
            for min_singular in list_min_singular:
                plt.plot(x_axis,list_train_loss_curve[n], label="({}, {})".format(i_dim - zero_dim, min_singular))
                n+=1

        #plt.title("Validation loss")
        plt.xlabel("epoch")
        plt.ylabel("train loss")

        plt.legend()
        plt.savefig("{}/test_ms_high_train.png".format(dirname))
        plt.clf()
        plt.close()

        logging.info(list_zero_dim)
        logging.info(list_min_singular)
        fde_ipn_estimated_rank_curve = np.asarray(list_estimated_rank_curve).reshape( num_zero_dim, num_min_singular, -1, num_thres)
        ### Consider last result
        fde_ipn_estimated_rank = fde_ipn_estimated_rank_curve[:,:,-1,:]


        x_axis = list_min_singular
        for i in range(num_zero_dim):
            plt.figure()
            zero_dim = list_zero_dim[i]
            true_rank = i_dim - zero_dim
            ### plot true_rank
            plt.plot(x_axis, (true_rank)*np.ones(len(x_axis)), linestyle="--", color="black")

            for k in range(num_thres):
                plt.plot(x_axis,fde_ipn_estimated_rank[i,:,k], label="FDE-IPN: thres= {}".format(list_zero_thres[k]), marker="+")
            if VS_VBMF:
                plt.plot(x_axis, vbmf_estimated_rank[i], label="VBMF", marker="+")


            plt.title("True rank = {}".format(true_rank))
            plt.xlabel("$\lambda_{min}$")
            plt.ylim(0,i_dim)
            plt.ylabel("Estimated rank")
            plt.legend()
            plt.savefig("{}/true_ranks-{}.png".format(dirname, true_rank ))
            plt.clf()
            plt.close()


    else: sys.exit(-1)


timer = Timer()
timer.tic()
test_sc(jobname =jobname, SUBO=True, VS_VBMF=i_vbmf)
timer.toc()
logging.info("total_time= {} min.".format(timer.total_time/60.))
