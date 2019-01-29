import numpy as np
import scipy as sp

from matrix_util import *
from random_matrices import *
import matplotlib.pyplot as plt
from timer import Timer
import os
import time
import logging
from tqdm import tqdm, trange

from cw import *
import copy

from utils.schedulers import *
from optimizers.adam import Adam
from optimizers.momentum import Momentum
from utils.samplers import *

TEST_C2 = False ###Use \C^2-valued subordination for test
BASE_C2 =   True ###Use \C^2-valued suborndaiton as BASE
if TEST_C2:
    import spn_c2
    from spn import *
else:
    if BASE_C2:
        from spn_c2 import *
    else:
        from spn import *

i_dpi = 120  #Resolution of figures
i_ext = "png"

def get_minibatch(sample, minibatch_size, n, scale, num_cauchy_rv, \
SAMPLING="CHOICE", MIX = "DIAGONAL"):
    """
    Choose minibatch from sample
    iter_per_epoch = len(sample)/minibatch_size
    """
    if SAMPLING == "SHUFFLE":
        mb_idx = n % minibatch_size
        if mb_idx == 0:
            np.random.shuffle(sample)
        mini = sample[minibatch_size*mb_idx:minibatch_size*(mb_idx+1)]
    elif SAMPLING == "CHOICE":

        mini = np.random.choice(sample, minibatch_size)

    else: sys.exit("SAMPLING is SHUFFLE or CHOICE")

    if MIX == "SEPARATE":
        new_mini = np.zeros(minibatch_size*num_cauchy_rv)
        c_noise =  sp.stats.cauchy.rvs(loc=0, scale=scale, size=num_cauchy_rv)
        n = 0
        for j in range(minibatch_size):
            for k in range(num_cauchy_rv):
                new_mini[n] = mini[j] + c_noise[k]
                n+=1
        return new_mini

    elif MIX == "X_ORIGIN":
        new_mini = np.zeros((minibatch_size,num_cauchy_rv))
        for i in range(minibatch_size):
            c_noise =  sp.stats.cauchy.rvs(loc=0, scale=scale, size=num_cauchy_rv)
            for j  in range(num_cauchy_rv):
                new_mini[i][j] = mini[i] + c_noise[j]
        new_mini = new_mini.flatten()
        mini = np.sort(new_mini)
        mini = np.array(mini, dtype=np.complex128)
        return mini
    elif MIX == "DIAGONAL":
        new_mini = np.zeros(minibatch_size)
        for i in range(minibatch_size):
            c_noise =  sp.stats.cauchy.rvs(loc=0, scale=scale, size=minibatch_size)
            new_mini[i] = mini[i] + c_noise[i]
        return new_mini


def get_learning_rate(idx, base_lr, lr_policy,  **kwards):
    assert base_lr >= 0
    assert idx >= 0
    if lr_policy == "inv":
        gamma = kwards["gamma"]
        power = kwards["power"]
        lr = base_lr * (1 + gamma * idx )**(- power)
        return lr
    elif lr_policy == "step":
        stepvalue = kwards["stepvalue"]
        num_step = len(stepvalue)
        for i in range(num_step):
            if idx < stepvalue[i]:
                return base_lr*kwards["decay"]**i


    elif lr_policy == "fix":
        return base_lr


    else:sys.exit()



def train_cw(dim, p_dim, sample,\
 base_scale = 0.01, num_cauchy_rv=4, base_lr = 0.1,minibatch_size=1,\
 max_epoch=20,  normalize_sample = False,\
 monitor_validation=True, test_b=-1):
    ###sample
    sample_size = len(sample)
    lambda_plus = (1 + sp.sqrt(p_dim/dim))**2
    ### SGD
    #base_lr = 0.1
    iter_per_epoch = int(sample_size/minibatch_size)
    max_iter = max_epoch*iter_per_epoch

    stepsize = -1 #iter_per_epoch
    momentum = 0#0.9
    clip_grad = -1

    optimizer = "Adam"
    lr_policy = "fix"

    if optimizer == "momentum":
        momentum = 0
        #momentum = momentum*np.ones(size+1)
        #momentum[-1] = 0.1  #momentum for sigma
        ###TODO test

    elif optimizer == "Adam":
        alpha = base_lr ###1e-4
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        lr = base_lr
    else: sys.exit("optimizer is momentum or Adam")




    if lr_policy == "inv":
        lr_kwards = {
        "gamma": 1e-4,
        "power": 0.75,
        }
    elif lr_policy == "step":
        lr_kwards = {
        "decay": 0.5,
        "stepvalue": [640, max_iter]
        }
    elif lr_policy == "fix":
        lr_kwards = dict()
    else:
        sys.exit("lr_policy is inv, step or fix")


    ### tf_summary, logging and  plot
    log_step = 20*iter_per_epoch
    plot_stepsize = log_step
    plot_stepsize = -1 #TODO Comment out this line for plotting density.

    ### inputs to be denoised
    sample = np.array(sample)
    if normalize_sample:
        max_row_sample = max(abs(sample))
        normalize_ratio = max(1., max_row_sample)
        sample /= max_row_sample
    else:
        normalize_ratio = 1.

    max_sample = np.max(abs(sample))
    #edge = max_sample/2.
    edge = 1.0
    ### initialization of weights
    ### TODO find nice value
    init_mean_b = 0.

    #b = init_mean_b+ np.random.uniform(p_dim)/sp.sqrt(p_dim)
    b = np.random.uniform(low=-1, high=1, size=p_dim)/sp.sqrt(p_dim)
    b = np.sort(b)
    b = np.array(b, dtype=np.complex128)


    logging.info("base_scale = {}".format(base_scale) )
    logging.info("num_cauchy_rv = {}".format(num_cauchy_rv) )
    logging.info("base_lr = {}".format(base_lr) )
    logging.info("minibatch_size = {}".format(minibatch_size) )

    logging.info("Initial b=\n{}".format(b))


    ### for monitoring
    if monitor_validation:
        n_test_b = test_b/normalize_ratio

    cw = CompoundWishart(dim=dim,p_dim=p_dim, minibatch=minibatch_size, scale=base_scale)
    cw.b = b
    if plot_stepsize > 0 and monitor_validation:
        x_axis = np.linspace(-max(sample)*4, max(sample)*4, 201) ## Modify
        logging.info("Plotting initial density...")
        ### Another cw for plotting.
        cw_for_plot = CompoundWishart(dim=dim, p_dim=p_dim, scale=base_scale)
        cw_for_plot.b = n_test_b
        y_axis_truth = cw_for_plot.density(x_axis)

        y_axis_init = cw.density(x_axis)
        max_y = max(  max(y_axis_init), max(y_axis_truth))+0.1
        #plt.plot(x_axis,y_axis_init, label="Init")
        plt.clf()
        plt.close()
        plt.figure()
        plt.rc('font', family="sans-serif", serif='Helvetica')
        plt.rc('text', usetex=True)
        plt.rcParams["font.size"] = 16
        #plt.rc("text", fontsize=12)
        plt.title("Perturbed single shot ESD  and $\gamma$-slice")
        plt.ylim(0, max_y)

        sample_for_plot = cw_for_plot.ESD(num_shot=1, num_cauchy_rv=200, COMPLEX=False)
        plt.hist(sample_for_plot, range=(min(x_axis), max(x_axis)), bins=200, normed=True,\
         label="perturbed \n single shot ESD",color="pink")


        plt.plot(x_axis,y_axis_truth, label="true $\gamma$-slice", linestyle="--", color="red")

        plt.legend()
        dirname = "../images/train_rnn_cw"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        plt.savefig("{}/plot_density_init.{}".format(dirname,i_ext), dpi=i_dpi)
        plt.clf()
        plt.close()
        logging.info("Plotting initial density...done.")


    scale = base_scale
    lr = base_lr
    train_loss_list = []
    val_loss_list = []



    ### local variables
    step_idx = 0
    average_loss = 0
    average_val_loss = 0
    average_b = 0
    cw.b = b


    if optimizer == "momentum":
        lr = base_lr
        old_grads = np.zeros(p_dim)
    elif optimizer == "Adam":
        mean_adam = np.zeros(p_dim)
        var_adam = np.zeros(p_dim)



    ### SGD
    for n  in trange(max_iter):
        ### learning rate
        #temp_idx =  int(n/iter_per_epoch) * iter_per_epoch
        temp_idx = n
        mini = get_minibatch(sample, minibatch_size, n, scale, num_cauchy_rv, SAMPLING="CHOICE")


        ### run
        cw.scale = scale
        cw.b = b
        new_grads, new_loss = cw.grad_loss(mini)
        train_loss_list.append(new_loss)


        ### learning rate
        if optimizer == "momentum":
            lr = get_learning_rate(n, base_lr, lr_policy, **lr_kwards)
            m_grads = lr*new_grads + momentum*old_grads


            ### TODO Delete after estimation
            #ad_momentum = 1- (1-momentum)*lr/base_lr
            #m_grads = lr*new_grads + ad_momentum*old_grads
            ###

        elif optimizer == "Adam":
            mean_adam = beta1 * mean_adam + ( 1- beta1)*new_grads
            var_adam = beta2 * var_adam + ( 1- beta2)*new_grads**2
            m = mean_adam/(1-beta1**(n+1))
            v = var_adam/(1-beta2**(n+1))
            m_grads = alpha*m/(sp.sqrt(v)+eps)


        logging.debug("m_grads=\n{}".format(m_grads))

        ### clip
        if clip_grad > 0:
            for k in range(p_dim):
                if abs(m_grads[k]) > clip_grad :
                    logging.info("m_grads[{}]={} is clipped.".format(k,m_grads[k]))
                    m_grads[k] = np.sign(m_grads[k])*clip_grad

        ### update
        old_b = np.copy(b)
        b = b - m_grads
        for k in range(p_dim):
            if abs(b[k]) > edge:
                logging.info( "b[{}]={} reached the boundary".format(k,b[k]))
                b[k] = sp.sign(b[k])*(edge - 1e-3)
                m_grads[k] = 0.


        ######################################
        ### Monitoring
        #####################################
        if monitor_validation:
            val_loss=np.linalg.norm(np.sort(b) - np.sort(n_test_b)) #+  np.abs(np.abs(sigma) - n_test_sigma)
            val_loss_list.append(val_loss)
            average_val_loss += val_loss
        average_loss += new_loss
        average_b += np.sort(b)
        if (n % log_step + 1 ) == log_step:
            average_loss /= log_step
            average_val_loss /= log_step
            average_b /= log_step
            logging.info("{0}/{4}-iter:lr = {1:4.3e}, scale = {2:4.3e}, cauchy = {3}, mini = {5}".format(n+1,lr,scale,num_cauchy_rv,max_iter, minibatch_size ))
            logging.info("train loss= {}".format( average_loss))
            if monitor_validation:
                logging.info("val_loss= {}  / average".format(average_val_loss))
            logging.debug("average-sorted(b) =\n{}  / average".format(average_b))

            average_loss = 0
            average_val_loss = 0
            average_b = 0

        logging.debug( "{}-iter:lr={},scale{}, loss: {}".format(n+1,lr, scale, new_loss) )
        logging.debug( "val_loss: {}".format(val_loss))


        if (n % plot_stepsize + 1 )== plot_stepsize:
            logging.info("Plotting density...")
            plt.clf()
            plt.close()
            plt.figure()
            plt.rc('font', family="sans-serif", serif='Helvetica')
            plt.rc('t{}', usetex=True)
            plt.rcParams["font.size"] = 16


            plt.title("Optimization: {} iteration".format(n))
            plt.ylim(0, max_y)

            if monitor_validation:
                #plt.plot(x_axis,y_axis_truth, label="$\gamma$-slice of true DE", color="red", linestyle="--")
                plt.plot(x_axis,y_axis_truth, color="red", linestyle="--")

            #plt.plot(x_axis,y_axis_init, label="Init")
            y_axis = cw.density(x_axis)
            #plt.legend()
            #plt.plot(x_axis,y_axis,label="{} iteration".format(n), color="blue")
            plt.plot(x_axis,y_axis, color="blue")

            plt.savefig("{}/plot_density_{}-iter".format(dirname, n+1),dpi=i_dpi)
            plt.clf()
            plt.close()
            logging.info("Plotting density...done")
    b *= normalize_ratio
    logging.info("result:")
    logging.info("b:\\{}".format(b))


    train_loss_array = np.array(train_loss_list)
    forward_iter = cw.forward_iter/ ( max_iter * minibatch_size)
    del cw
    result = dict(b= b, train_loss= train_loss_array, forward_iter= forward_iter)
    if monitor_validation:
        val_loss_array = np.array(val_loss_list)
        result["val_loss"] = val_loss_array
    return result
