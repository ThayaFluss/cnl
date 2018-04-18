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
 base_scale = 2e-1, dim_cauchy_vec=2, base_lr = 0.1,minibatch_size=1,\
 max_epoch=120, normalize_sample = False,\
 reg_coef = 2e-4,\
 monitor_validation=True, monitor_KL=False, test_diag_A=-1, test_sigma=-1, \
 list_zero_thres=[1e-5,1e-4,1e-3,1e-2,1e-1], SUBO=True,  stop_for_rank=False):
    update_sigma = True
    add_noise_for_saddle = True

    if np.allclose(test_diag_A, -1) or np.allclose(test_sigma, -1):
        monitor_validation = False
    ### param cauchy noise
    #base_scale = 0.01
    #dim_cauchy_vec = 4

    ###sample
    #minibatch_size = 1
    sample_size =sample.shape[0]
    ### SGD
    #base_lr = 0.1
    iter_per_epoch = int(sample_size/minibatch_size)
    max_iter = max_epoch*iter_per_epoch
    stepsize = -1 #iter_per_epoch
    momentum = 0#0.9
    REG_TYPE = "L1"
    lr_policy = "inv"
    if lr_policy == "inv":
        stepsize = iter_per_epoch
        gamma = 1e-4*dim
        power = 0.75
        scale_gamma = 0
        scale_power = 0.75
    elif lr_policy == "step":
        decay = 0.5
        scale_decay =1.
        stepvalue = [640]
        #stepvalue = [int(max_iter/2)]
        stepvalue.append(max_iter)
        logging.info("stepvalue= {}".format(stepvalue))


    ### tf_summary, logging and  plot
    log_step = dim
    stdout_step = log_step*10
    KL_log_step = 10*dim
    plot_stepsize = stdout_step
    plot_stepsize = -1
    stop_count_thres = 100


    ### update sigma
    lr_mult_sigma = 0.
    if update_sigma:
        #momentum = momentum*np.ones(size+1)
        #momentum[-1] = 0.1  #momentum for sigma
        start_update_sigma = 0
        lr_mult_sigma = 1./dim


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
        plt.savefig("{}/plot_density_init.png".format(dirname),dpi=i_dpi)
        plt.clf()
        plt.close()
        logging.info("Plotting initial density...done.")


    scale = base_scale
    lr = base_lr
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
    #list_zero_thres.append(sc.sigma)
    num_zero = np.zeros(len(list_zero_thres))
    old_PA = np.zeros(dim)
    old_Psigma = 0
    stop_count = 0
    ### SGD
    for n in trange(1,max_iter+1):
        ### learning rate
        if lr_policy == "inv":
            if n % stepsize == 0:
                lr = base_lr * (1 + gamma * float(n)/stepsize)**(- power)
                scale = base_scale * (1 + scale_gamma * float(n)/stepsize)**(- scale_power)
        elif lr_policy == "step":
            step= stepvalue[step_idx]
            if n > step:
                step_idx += 1
            lr =  base_lr*decay**(step_idx)
            scale = base_scale*scale_decay**step_idx


        ### minibatch
        n_grp = n % iter_per_epoch
        if  n_grp == 0:
            #logging.info("shuffle sample")
            np.random.shuffle(sample)
        mini = sample[minibatch_size*n_grp:minibatch_size*(n_grp+1)]
        mini = np.sort(mini)
        new_mini = np.zeros((minibatch_size,dim_cauchy_vec))
        for i in range(minibatch_size):
            c_noise =  sp.stats.cauchy.rvs(loc=0, scale=scale, size=dim_cauchy_vec)
            for j  in range(dim_cauchy_vec):
                new_mini[i][j] = mini[i] - c_noise[j]
        new_mini = new_mini.flatten()
        mini = np.sort(new_mini)
        mini = np.array(mini, dtype=np.complex128)



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

        r_grads, r_loss =  sc.regularization_grad_loss(diag_A, sigma,reg_coef=reg_coef, TYPE=REG_TYPE)
        new_grads += r_grads
        new_loss += r_loss
        train_loss_list.append(new_loss)

        if add_noise_for_saddle:
            import pdb; pdb.set_trace()
            noise_for_saddle = np.random.randn(new_grads.shape[0])
            noise_for_saddle /= np.norm(noise_for_saddle)
            new_grads += noise_for_saddle

        new_PA = new_grads[:-1]
        new_Psigma = new_grads[-1]


        #logging.info("new_Psigma:{}".format(new_Psigma))
        #logging.info("new_PA:\n{}".format(new_PA))
        m_Psigma = lr*new_Psigma + momentum*old_Psigma
        m_Psigma *= lr_mult_sigma
        m_PA = lr*new_PA + momentum*old_PA
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
            if abs(sigma**2) > edge:
                logging.info("sigma reached the boundary:{}".format(sigma))
                sigma  = sp.sqrt(edge) - 1e-2
                m_Psigma = -1e-8
        old_m_PA= np.copy(m_PA)
        old_Psigma = np.copy(m_Psigma)


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
                    logging.info("val_loss_average= {}".format(val_loss_average))
            if n % stdout_step == 0:

                logging.info("sigma= {}".format(average_sigma))
                #logging.info("diag_A (sorted)=\n{}  / 10-iter".format(np.sort(average_diagA)))
                #logging.info( "diag_A (raw) =\n{}".format(diag_A))
                logging.info("num_zero={} / thres={}".format(num_zero,list_zero_thres))
                logging.debug("average-sorted-abs(diag_A) =\n{}".format(average_diagA))
                #logging.info("diag_A/ average: min, mean, max= \n {}, {}, {} ".format(np.min(average_diagA), np.average(average_diagA), np.max(average_diagA)))

            if monitor_KL and n % (KL_log_step) == 0:
                logging.info("Computing KL divergence from truth ...")
                sc.diag_A = average_diagA
                sc.sigma = average_sigma
                cross_entropy = sc.loss(sc.ESD(num_shot=num_shot_for_KL, dim_cauchy_vec=dim_cauchy_vec_for_KL))
                KL = cross_entropy - entropy
                logging.info("val_KL= : {}".format(KL))


            if n < max_iter  :
                average_loss = 0
                average_val_loss = 0
                average_sigma = 0
                average_diagA = 0


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
    del sc
    if monitor_validation:
        del sc_for_plot
    return average_diagA, average_sigma, train_loss_array, val_loss_array, num_zero_array


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
    lr_policy = "inv"
    if lr_policy == "inv":
        stepsize = 1 #iter_per_epoch
        gamma = 1e-4
        power = 0.75
        scale_gamma = 0
        scale_power = 0.75
    elif lr_policy == "step":
        decay = 0.5
        scale_decay =1.
        stepvalue = [640]
        #stepvalue = [int(max_iter/2)]
        stepvalue.append(max_iter)
        logging.info("stepvalue= {}".format(stepvalue))


    ### tf_summary, logging and  plot
    log_step = 20*iter_per_epoch
    plot_stepsize = log_step
    #plot_stepsize = -1 #TODO Comment out this line for plotting density.

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
    edge = 0.6
    ### initialization of weights
    ### TODO find nice value
    init_mean_b = 0.

    b = init_mean_b+ np.random.randn(p_dim)/sp.sqrt(p_dim)
    #b = random.uniform(low=-1, high=1, sizep_dim)
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
        plt.rc('font', family='serif', serif='Times')
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
        plt.savefig("{}/plot_density_init.png".format(dirname), dpi=i_dpi)
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
    old_grads = np.zeros(p_dim)
    cw.b = b
    ### SGD
    for n in trange(max_iter):
        ### learning rate
        if lr_policy == "inv":
            if n % stepsize == 0:
                lr = base_lr * (1 + gamma * float(n)/stepsize)**(- power)
                scale = base_scale * (1 + scale_gamma * float(n)/stepsize)**(- scale_power)
        elif lr_policy == "step":
            step= stepvalue[step_idx]
            if n > step:
                step_idx += 1
            lr =  base_lr*decay**(step_idx)
            scale = base_scale*scale_decay**step_idx


        ### minibatch
        n_grp = n % iter_per_epoch
        if  n_grp == 0:
            #logging.info("shuffle sample")
            np.random.shuffle(sample)
        mini = sample[minibatch_size*n_grp:minibatch_size*(n_grp+1)]
        mini = np.sort(mini)
        c_noise =  sp.stats.cauchy.rvs(loc=0, scale=scale, size=dim_cauchy_vec)
        new_mini = np.zeros((minibatch_size,dim_cauchy_vec))
        for i in range(minibatch_size):
            for j  in range(dim_cauchy_vec):
                new_mini[i][j] = mini[i] - c_noise[j]
        new_mini = new_mini.flatten()
        mini = np.sort(new_mini)

        mini = np.array(mini, dtype=np.complex128)



        ### run
        cw.scale = scale
        cw.b = b
        new_grads, new_loss = cw.grad_loss(mini)
        train_loss_list.append(new_loss)

        #logging.info("new_Psigma:{}".format(new_Psigma))
        #logging.info("new_PA:\n{}".format(new_PA))
        ad_momentum = 1- (1-momentum)*lr/base_lr
        m_grads = lr*new_grads + ad_momentum*old_grads
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
            plt.rc('font', family='serif', serif='Times')
            plt.rc('text', usetex=True)
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
    if monitor_validation:
        val_loss_array = np.array(val_loss_list)
        return b, train_loss_array, val_loss_array
    else:
        return b, train_loss_array, -1
