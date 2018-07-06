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

from fde_cw import *

TEST_C2 = False ###Use \C^2-valued subordination
BASE_C2 =   True ###Use \C^2-valued suborndaiton as BASE
if TEST_C2:
    import fde_sc_c2
    from fde_sc import *
else:
    if BASE_C2:
        from fde_sc_c2 import *
    else:
        from fde_sc import *

i_dpi = 120  #Resolution of figures
i_ext = "png"

def get_minibatch(sample, minibatch_size, n, scale, dim_cauchy_vec, \
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
        ###TODO temp
        if MIX == "DIAGONAL":
            minibatch_size = dim_cauchy_vec

        mini = np.random.choice(sample, minibatch_size)

    else: sys.exit("SAMPLING is SHUFFLE or CHOICE")

    if MIX == "SEPARATE":
        new_mini = np.zeros(minibatch_size*dim_cauchy_vec)
        c_noise =  sp.stats.cauchy.rvs(loc=0, scale=scale, size=dim_cauchy_vec)
        n = 0
        for j in range(minibatch_size):
            for k in range(dim_cauchy_vec):
                new_mini[n] = mini[j] + c_noise[k]
                n+=1
        return new_mini

    elif MIX == "X_ORIGIN":
        new_mini = np.zeros((minibatch_size,dim_cauchy_vec))
        for i in range(minibatch_size):
            c_noise =  sp.stats.cauchy.rvs(loc=0, scale=scale, size=dim_cauchy_vec)
            for j  in range(dim_cauchy_vec):
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
    elif lr_policy == "fix":
        return base_lr




def KL_divergence(diag_A,sigma, sc_true, num_shot = 20, dim_cauchy_vec=100):
    sc = SemiCircular(dim = np.shape(diag_A)[0], scale = sc_true.scale)
    sc.diag_A = diag_A
    sc.sigma = sigma
    #sc_true = SemiCircular(diag_A = a_true, sigma=sigma_true)
    num_shot = 10
    dim_cauchy_vec = 10
    #sample = sc.ESD(num_shot, dim_cauchy_vec)
    sample_true = sc_true.ESD(num_shot=num_shot, dim_cauchy_vec=dim_cauchy_vec)
    timer = Timer()
    timer.tic()
    entropy = sc_true.loss_subordination(sample_true)
    timer.toc()

    logging.info("entropy= {}, time= {}".format(entropy, timer.total_time))

    timer = Timer()
    timer.tic()
    cross_entropy = sc.loss_subordination(sample_true)
    timer.toc()
    logging.info("cross_entropy= {}, time= {}".format(cross_entropy, timer.total_time))

    KL = cross_entropy - entropy
    return KL


def train_fde_sc(dim, p_dim, sample,\
 base_scale = 1e-1, dim_cauchy_vec=1, base_lr = 1e-4,minibatch_size=1,\
 max_epoch=120, normalize_sample = False,\
 reg_coef = 1e-3,\
 monitor_validation=True, monitor_KL=False, test_diag_A=-1, test_sigma=-1, \
 list_zero_thres=[1e-5,1e-4,1e-3,1e-2,1e-1], SUBO=True,  stop_for_rank=False):
    update_sigma = True

    if np.allclose(test_diag_A, -1) or np.allclose(test_sigma, -1):
        monitor_validation = False
    ### param cauchy noise
    #base_scale = 0.01
    #dim_cauchy_vec = 4

    ###sample
    #minibatch_size = 1
    sample_size =len(sample)
    assert sample_size == sample.shape[0]
    ### SGD
    #base_lr = 0.1
    iter_per_epoch = int(sample_size/minibatch_size)
    max_iter = max_epoch*iter_per_epoch
    stepsize = -1 #iter_per_epoch

    ### for online L1 -regularization
    ### LLZ, Sparse Online Learning via Truncated Gradient
    ### Depricated now
    use_truncated_grad = False
    truncate_step = 1

    optimizer = "Adam"
    if optimizer == "momenutm":
        momentum = 0
        #momentum = momentum*np.ones(size+1)
        #momentum[-1] = 0.1  #momentum for sigma
        start_update_sigma = 0
        lr_mult_sigma = 1./dim
        ###TODO test
        #lr_mult_sigma = 1./sp.sqrt(dim)
        #lr_mult_sigma = 0.1

    elif optimizer == "Adam":
        alpha = base_lr
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        lr = base_lr
    else: sys.exit("optimizer is momentum or Adam")

    if update_sigma:
        start_update_sigma = 0
    else:
        lr_mult_sigma = 0.


    REG_TYPE = "L1"
    lr_policy = "inv"
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
    log_step = iter_per_epoch
    stdout_step = log_step*10
    KL_log_step = 10*dim
    plot_stepsize = stdout_step
    plot_stepsize = -1
    stop_count_thres = 100



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
    #edge = max_sq_sample*1.1
    edge = 1.2
    clip_grad = 100

    ### initialization of weights
    ### TODO find nice value
    init_mean_diag_A =np.average(sq_sample)/2
    #init_mean_diag_A = 0.5
    #sigma =  0.1*init_mean_diag_A
    if monitor_validation and not update_sigma:
        sigma = test_sigma
    else:
        sigma = 0.2
    sigma = abs(sigma)

    diag_A = sq_sample
    diag_A = abs(diag_A)
    diag_A = np.sort(diag_A)
    diag_A = np.array(diag_A, dtype=np.complex128)



    logging.info("base_scale = {}".format(base_scale) )
    logging.info("dim_cauchy_vec = {}".format(dim_cauchy_vec) )
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

        if monitor_KL:
            num_shot_for_KL = 10
            dim_cauchy_vec_for_KL = 50
            logging.info("computing entropy ...")
            timer = Timer()
            timer.tic()
            temp_sample =sc_for_plot.ESD(num_shot=num_shot_for_KL, dim_cauchy_vec=dim_cauchy_vec_for_KL)
            entropy = sc_for_plot.loss(temp_sample)
            timer.toc()
            logging.info("entropy of true= {}, time= {} sec".format(entropy, timer.total_time))



    sc = SemiCircular(dim=dim,p_dim=p_dim, scale=base_scale)
    sc.set_params(diag_A, sigma)
    if TEST_C2:
        sc2 = fde_sc_c2.SemiCircular(dim=dim,p_dim=p_dim, scale=base_scale)
        sc2.set_params(diag_A, sigma)


    if plot_stepsize > 0:
        x_axis = np.linspace(-max(sample)*4, max(sample)*4, 201) ## Modify
        logging.info("Plotting initial density...")
        if monitor_validation:
            y_axis_truth = sc_for_plot.density_subordinaiton(x_axis)

            sample_for_plot =  sc_for_plot.ESD(num_shot=1,dim_cauchy_vec=200)

            plt.hist(sample_for_plot, range=(min(x_axis), max(x_axis)), bins=100, normed=True, label="sampling from a true model \n perturbed by Cauchy($0,\gamma$)",color="pink")

            plt.plot(x_axis,y_axis_truth, linestyle="--", label="true $\gamma$-slice", color="red")







        sc.set_params(diag_A, sigma)
        y_axis_init = sc.density_subordinaiton(x_axis)
        plt.plot(x_axis,y_axis_init, label="Init")
        plt.legend()
        dirname = "images/train_rnn_sc"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        plt.savefig("{}/plot_density_init.{}".format(dirname,i_ext),dpi=i_dpi)
        plt.clf()
        plt.close()
        logging.info("Plotting initial density...done.")


    scale = base_scale
    ### to return
    train_loss_list = []
    val_loss_list = []
    num_zero_list = []

    ### local variables
    step_idx = 0
    average_loss = 0
    average_val_loss = 0
    average_sigma = 0
    average_diagA = 0

    o_zero_thres = list_zero_thres


    if optimizer == "momentum":
        lr = base_lr
        old_grads = np.zeros(dim+1)
    elif optimizer == "Adam":
        mean_adam = np.zeros(dim + 1)
        var_adam = np.zeros(dim + 1)


    old_forward_iter = np.zeros(1)
    stop_count = 0
    ### SGD
    for n in trange(max_iter):
        ### for epoch base
        # e_idx = int(n/ iter_per_epoch)
        mini = get_minibatch(sample, minibatch_size, n, scale, dim_cauchy_vec, SAMPLING="CHOICE")

        ### TODO to add sigma to zero thres
        #list_zero_thres = o_zero_thres + [sigma]
        list_zero_thres = o_zero_thres
        num_zero = np.zeros(len(list_zero_thres))

        ### run
        sc.scale= scale
        if SUBO:
            sc.update_params(diag_A, sigma)
            new_grads, new_loss = sc.grad_loss_subordination(mini)

            if TEST_C2:
                sc2.scale = scale
                sc2.update_params(diag_A, sigma)
                sc2_new_grads, sc2_new_loss = sc2.grad_loss_subordination(mini)
                print("")
                print("sub-Psigma:", sc2_new_grads[-1] - new_grads[-1])
                print("sc2 psigma:", sc2_new_grads[-1])
                print("psigma:", new_grads[-1])
                #print("subloss:",sc2_new_loss - new_loss)
                #import pdb; pdb.set_trace()



            #t_new_grads, t_new_loss = sc.grad_loss(diag_A, sigma, mini)
            #logging.debug("test_loss: {}".format(np.linalg.norm(new_loss- t_new_loss)))
            #logging.debug("test_grad: {}".format(np.linalg.norm(new_grads- t_new_grads)))
        else :
            new_grads, new_loss = sc.grad_loss(diag_A, sigma, mini)


        if not use_truncated_grad:
            r_grads, r_loss =  sc.regularization_grad_loss(diag_A, sigma,reg_coef=reg_coef, TYPE=REG_TYPE)
            new_grads += r_grads
            new_loss += r_loss
        train_loss_list.append(new_loss)



        if optimizer == "momentum":
            ### learning rate
            lr = get_learning_rate(n, base_lr, lr_policy, **lr_kwards)
            m_grads = lr*new_grads + momentum*old_grads

            m_grads[-1] *= lr_mult_sigma ### for rescale new_Psigma


        elif optimizer == "Adam":
            mean_adam = beta1 * mean_adam + ( 1- beta1)*new_grads
            var_adam = beta2 * var_adam + ( 1- beta2)*new_grads**2
            m = mean_adam/(1-beta1**(n+1))
            v = var_adam/(1-beta2**(n+1))
            m_grads = m*alpha/(sp.sqrt(v)+eps)
            lr_for_trunc = alpha/(sp.sqrt(v)+eps)



        m_PA = m_grads[:-1]
        m_Psigma = m_grads[-1]



        logging.debug("m_Psigma=\n{}".format(m_Psigma))
        logging.debug("m_PA=\n{}".format(m_PA))

        ### clip
        if clip_grad > 0:
            for k in range(dim):
                if abs(m_PA[k]) > clip_grad :
                    logging.info("m_PA[{}]={} is clipped.".format(k,m_PA[k]))
                    m_PA[k] = np.sign(m_PA[k])*clip_grad
            if abs(m_Psigma) > clip_grad :
                logging.info("m_Psigma={} is clipped.".format(m_Psigma))
                m_Psigma = np.sign(m_Psigma)*clip_grad

        ### update
        old_diag_A = np.copy(diag_A)
        diag_A = diag_A - m_PA

        ### Online L1 reguralization
        if use_truncated_grad:
            if n > 0 and n % truncate_step == 0:
                for k in range(dim):
                    #lrk = lr_for_trunc[k]
                    lrk = alpha
                    ak = diag_A[k]
                    if ak  > 0 and ak  < reg_coef:
                        diag_A[k] = max(0, ak - lrk*reg_coef*truncate_step)
                    elif ak < 0 and ak < -reg_coef:
                        diag_A[k] = min(0, ak + lrk*reg_coef*truncate_step)





        for k in range(dim):
            if abs(diag_A[k]) > edge:
                logging.info( "diag_A[{}]={} reached the boundary".format(k,diag_A[k]))
                #import pdb; pdb.set_trace()

                diag_A[k] =edge  -  1e-2
                m_PA[k] = -1e-8


        if update_sigma:
            logging.debug( "m_Psigma=", m_Psigma)
            if n > start_update_sigma:
                sigma -= m_Psigma
            if abs(sigma) > edge:
                logging.info("sigma reached the boundary:{}".format(sigma))
                sigma  = sp.sqrt(edge) - 1e-2
                m_Psigma = -1e-8


        old_m_PA= np.copy(m_PA)
        old_m_Psigma = np.copy(m_Psigma)
        olg_grads = np.hstack((old_m_PA, old_m_Psigma))


        ######################################
        ### Monitoring
        #####################################
        if monitor_validation:
            #val_loss=np.sum(np.abs(np.sort(np.abs(diag_A)) - np.sort(np.abs(n_test_diag_A)))) +  np.abs(np.abs(sigma) - n_test_sigma)
            val_loss=np.linalg.norm(np.sort(np.abs(diag_A)) - np.sort(np.abs(n_test_diag_A))) +  np.abs(np.abs(sigma) - n_test_sigma)
            val_loss_list.append(val_loss)
            average_val_loss += val_loss

        num_zero_list.append( \
        [  np.where( np.abs(diag_A) < list_zero_thres[l])[0].size \
        for l in range(len(list_zero_thres))])

        average_loss += new_loss
        average_sigma += sigma
        average_diagA += np.sort(np.abs(diag_A))

        if n %  log_step == 0:
            ### Count zeros under several thresholds
            num_zero = num_zero_list[-1]

            if n > 0:
                average_loss /= log_step
                average_sigma /= log_step
                average_diagA /= log_step
                if monitor_validation:
                    average_val_loss /= log_step
                if stop_for_rank:
                    num_zero_old = num_zero_list[-log_step]
                    if np.allclose(num_zero,num_zero_old) and not np.allclose(num_zero, 0):
                            stop_count += 1
                    else:
                        stop_count = 0
                    if stop_count == stop_count_thres:
                        import pdb; pdb.set_trace()
                        break

            #average_sigma *= normalize_ratio
            #average_diagA *= normalize_ratio
            if n % stdout_step == 0:
                logging.info("{0}/{4}-iter:lr = {1:4.3e}, scale = {2:4.3e}, num_cauchy = {3}".format(n,lr,scale,dim_cauchy_vec,max_iter ))
                logging.info("loss= {}".format( average_loss))

            if monitor_validation:

                #val_loss_average = np.sum(np.abs(np.sort(np.abs(average_diagA)) - np.sort(np.abs(n_test_diag_A)))) \
                val_loss_average = np.linalg.norm(np.sort(np.abs(average_diagA)) - np.sort(np.abs(n_test_diag_A)))\
                +  np.abs(np.abs(average_sigma) - n_test_sigma)
                if n % stdout_step == 0:
                    logging.info("val_loss= {}".format(average_val_loss))
                    logging.debug("val_loss_average= {}".format(val_loss_average))
            if n % stdout_step == 0:

                logging.info("sigma= {}".format(average_sigma))
                #logging.info("diag_A (sorted)=\n{}  / 10-iter".format(np.sort(average_diagA)))
                #logging.info( "diag_A (raw) =\n{}".format(diag_A))
                logging.info("num_zero={} / thres={}".format(num_zero,list_zero_thres))
                logging.debug("average-sorted-abs(diag_A) =\n{}".format(average_diagA))
                #logging.info("diag_A/ average: min, mean, max= \n {}, {}, {} ".format(np.min(average_diagA), np.average(average_diagA), np.max(average_diagA)))
                forward_iter = sc.forward_iter
                diff_forward_iter = forward_iter - old_forward_iter
                diff_forward_iter = diff_forward_iter/ stdout_step
                logging.info("forward_iter={}".format(diff_forward_iter))
                old_forward_iter[0] = forward_iter[0]
            if monitor_KL and n % (KL_log_step) == 0:
                logging.info("Computing KL divergence from truth ...")
                sc.diag_A = average_diagA
                sc.sigma = average_sigma
                cross_entropy = sc.loss(sc.ESD(num_shot=num_shot_for_KL, dim_cauchy_vec=dim_cauchy_vec_for_KL))
                KL = cross_entropy - entropy
                logging.info("val_KL= : {}".format(KL))



        logging.debug( "{}-iter:lr={},scale{}, loss: {}".format(n,lr, scale, new_loss) )
        logging.debug( "sigma: {}".format(sigma))
        #diag_A=np.sort(diag_A)
        if monitor_validation:
            logging.debug( "val_loss: {}".format(val_loss))


        logging.debug( "mean_of_diagA: {}, var_of_A={}".format(np.average(diag_A), np.var(diag_A) ))
        logging.debug( "diag_A (raw) =\n{}".format(diag_A))



        if plot_stepsize > 0 and n % plot_stepsize == 0:
            logging.info("Plotting density...")
            plt.clf()
            plt.close()
            plt.figure()
            if monitor_validation:

                y_axis_truth = sc_for_plot.density_subordinaiton(x_axis)
                plt.plot(x_axis,y_axis_truth, label="truth", linestyle="--", color="red")
            #plt.plot(x_axis,y_axis_init, label="Init")
            sc.set_params(diag_A, sigma)
            y_axis = sc.density_subordinaiton(x_axis)
            plt.plot(x_axis,y_axis, label="{}-iter".format(n), color="blue")
            plt.legend()
            plt.savefig("{}/plot_density_{}-iter".format(dirname, n),dpi=i_dpi)
            plt.clf()
            plt.close()
            logging.info("Plotting density...done")


    average_loss /= log_step
    average_sigma /= log_step
    average_diagA /= log_step
    average_sigma *= normalize_ratio
    average_diagA *= normalize_ratio
    logging.info("result:")
    logging.info("sigma:".format(average_sigma))
    logging.info("diag_A:\\{}".format(average_diagA))
    train_loss_array = np.array(train_loss_list)
    if monitor_validation:
        val_loss_array = np.array(val_loss_list)

    else:
        val_loss_array = -1

    num_zero_array = np.asarray(num_zero_list)
    forward_iter = sc.forward_iter/ (max_iter*minibatch_size)

    del sc
    ### TODO for sigma
    #list_zero_thres = list_zero_thres[:-1]
    if monitor_validation:
        del sc_for_plot
    return average_diagA, average_sigma, train_loss_array, val_loss_array, num_zero_array, forward_iter


def train_fde_cw(dim, p_dim, sample,\
 base_scale = 0.01, dim_cauchy_vec=4, base_lr = 0.1,minibatch_size=1,\
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
    clip_grad = 100

    optimizer = "Adam"
    if optimizer == "momentum":
        momentum = 0
        #momentum = momentum*np.ones(size+1)
        #momentum[-1] = 0.1  #momentum for sigma
        start_update_sigma = 0
        lr_mult_sigma = 1./dim
        ###TODO test
        #lr_mult_sigma = 1./sp.sqrt(dim)
        #lr_mult_sigma = 0.1

    elif optimizer == "Adam":
        alpha = base_lr ###1e-4
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        lr = base_lr
    else: sys.exit("optimizer is momentum or Adam")




    lr_policy = "fix"
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
    logging.info("dim_cauchy_vec = {}".format(dim_cauchy_vec) )
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

        sample_for_plot = cw_for_plot.ESD(num_shot=1, dim_cauchy_vec=200, COMPLEX=False)
        plt.hist(sample_for_plot, range=(min(x_axis), max(x_axis)), bins=200, normed=True,\
         label="perturbed \n single shot ESD",color="pink")


        plt.plot(x_axis,y_axis_truth, label="true $\gamma$-slice", linestyle="--", color="red")

        plt.legend()
        dirname = "images/train_rnn_cw"
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
        mini = get_minibatch(sample, minibatch_size, n, scale, dim_cauchy_vec, SAMPLING="CHOICE")


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
        if n %  log_step == 0:
            if n > 0:
                average_loss /= log_step
                average_val_loss /= log_step
                average_b /= log_step
            logging.info("{0}/{4}-iter:lr = {1:4.3e}, scale = {2:4.3e}, num_cauchy = {3}".format(n,lr,scale,dim_cauchy_vec,max_iter ))
            logging.info("loss= {}  / average".format( average_loss))
            if monitor_validation:
                logging.info("val_loss= {}  / average".format(average_val_loss))
            logging.debug("average-sorted(b) =\n{}  / average".format(average_b))

            average_loss = 0
            average_val_loss = 0
            average_b = 0

        logging.debug( "{}-iter:lr={},scale{}, loss: {}".format(n,lr, scale, new_loss) )
        logging.debug( "val_loss: {}".format(val_loss))


        if plot_stepsize > 0 and n % plot_stepsize == 0:
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

            plt.savefig("{}/plot_density_{}-iter".format(dirname, n),dpi=i_dpi)
            plt.clf()
            plt.close()
            logging.info("Plotting density...done")
    b *= normalize_ratio
    logging.info("result:")
    logging.info("b:\\{}".format(b))


    train_loss_array = np.array(train_loss_list)
    forward_iter = cw.forward_iter/ ( max_iter * minibatch_size)
    del cw
    if monitor_validation:
        val_loss_array = np.array(val_loss_list)
        return b, train_loss_array, val_loss_array, forward_iter
    else:
        return b, train_loss_array, -1, forward_iter
