import numpy as np
import scipy as sp

from matrix_util import *
from random_matrices import *
import matplotlib.pyplot as plt
from timer import Timer
import os
import time
import logging

from multiprocessing import Pool
from multiprocessing import Process


#from cauchy import *
from tf_recurrent_cauchy_net import *

i_dim = 36
def monitor_optimization(dim,  test_diag_A,test_sigma,sample):

    update_sigma = False

    ### param Cauchy noise
    use_tf_seed = False  ###use tf's Cauchy distribution or taht of sp.stats
    base_scale = 0.1
    num_Cauchy_seeds = 1

    ###sample
    minibatch_size = 6
    sample_size = len(sample)
    normalize_sample = True
    ### SGD
    base_lr = 0.01
    max_epoch = 500
    iter_per_epoch = int(sample_size/minibatch_size)
    max_iter = max_epoch*iter_per_epoch
    stepsize = iter_per_epoch
    momentum = 0.9
    clip_grad = 0.5
    gamma = 1e-4
    power = 0.75
    scale_gamma = 1e-4
    scale_power = 0.75


    ### tf_summary, logging and  plot
    use_summary = False
    log_step = 10
    plot_stepsize = -1
    plot_stepsize = log_step*10

    ### decay step
    if stepsize < 0:
        stepvalue = [500,1000]
        stepvalue.append(max_iter)
        logging.info("stepvalue= {}".format(stepvalue))
    ### update sigma
    if update_sigma:
        #momentum = momentum*np.ones(size+1)
        #momentum[-1] = 0.1  #momentum for sigma
        start_update_sigma = 0
        reg_coef_sigma = 1



    ### inputs to be denoised
    sample = np.array(sample)
    if normalize_sample:
        max_row_sample = max(sample)
        normalize_ratio = sp.sqrt(max_row_sample)
        sample /= max_row_sample
    else:
        normalize_ratio = 1.

    sq_sample = sp.sqrt(sample)
    max_sq_sample = max(sq_sample)
    sample_var = np.var(sq_sample)

    ### initialization of weights
    sigma = 1. + 0.*1j
    sigma /= normalize_ratio
    init_mean_diag_A =np.average(sq_sample)

    #diag_A = init_mean_diag_A*1.1 +2*(np.random.random(dim) -0.5)*sample_var
    diag_A = init_mean_diag_A -0.3 + (np.random.random(dim) - 0.5)/sp.sqrt(dim)
    half_dim = int(dim/2)
    for d in range(half_dim):
        diag_A[d] = 0.2 + (np.random.random() - 0.5)/sp.sqrt(dim)
        diag_A[d+half_dim] = 0.9 + (np.random.random() - 0.5)/sp.sqrt(dim)
    diag_A = np.sort(diag_A)
    diag_A = abs(diag_A)
    diag_A = np.array(diag_A, dtype=np.complex128)

    logging.info("Initial diag_A=\n{}".format(diag_A))
    logging.info("Initial sigma={}".format(sigma))


    ### for monitoring
    n_test_diag_A = test_diag_A/normalize_ratio
    n_test_sigma = test_sigma/normalize_ratio

    sc = SemiCircular(eps=base_scale)
    if plot_stepsize > 0:
        x_axis = np.linspace(0.01, max(sample)*1.1, 201) ## Modify
        logging.info("Plotting initial density...")
        y_axis_truth = sc.square_density(x_axis,np.diag(n_test_diag_A), n_test_sigma)
        plt.plot(x_axis,y_axis_truth, label="Truth")
        y_axis_init = sc.square_density(x_axis,np.diag(diag_A), sigma)
        plt.plot(x_axis,y_axis_init, label="Init")
        plt.legend()
        plt.savefig("images/tf_test_optimize/plot_density_init")
        logging.info("Plotting initial density...done.")



    ###################################
    ####### tensorflow settings #######
    ###################################
    depth = 20
    depth_sc = 20
    omega_init = 1j*np.identity(2)
    G_sc_init = -1j*np.identity(2)
    ### initialize
    omega_op = tf.constant(omega_init, tf.complex128, name= "omega")
    G_sc_op = tf.constant( G_sc_init,tf.complex128,  name="G_sc")

    ### Variable
    diag_A_op = tf.Variable(diag_A, tf.complex128, name = "diag_A")
    sigma_op = tf.Variable( sigma, tf.complex128, name = "sigma")
    #temp = np.array([0.5, 1., 1.5], dtype=np.complex128)
    #minibatch_op = tf.Variable(temp, tf.complex128, name = "minibatch")


    ### placeholder
    #minibatch_op = tf.placeholder(tf.complex128, name = "minibatch")
    scale_op = tf.placeholder(tf.complex128, name = "scale_op")
    minibatch_op = tf.placeholder( tf.complex128, name = "minibatch")
    ### construct graph
    logging.info("Construct graph ...")
    loss, Psigma, PA = grad_Cauchy_noise_loss(dim, minibatch_op, minibatch_size, \
    num_Cauchy_seeds,scale_op, \
    diag_A_op, sigma_op, omega_op, G_sc_op, depth, depth_sc, use_tf_seed)
    logging.info("Construct graph ... ok")

    ### Session
    ### multi-threds
    sess = tf.InteractiveSession(config = tf.ConfigProto(
    intra_op_parallelism_threads=4 ))
    ### Summary
    if use_summary:
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('./tb_train', graph=sess.graph)
    ### Initialize
    tf.global_variables_initializer().run()

    ###
    comparing_with_dirct = True
    if comparing_with_dirct:
        sc = SemiCircular(update_sigma=update_sigma, minibatch=minibatch_size, eps=base_scale)


    ### local variables
    step_idx = 0
    average_loss = 0
    average_test_loss = 0
    average_sigma = 0
    average_diagA = 0
    old_PA = np.zeros(dim)
    old_Psigma = 0
    ### SGD
    for n in range(max_iter):
        ### learning rate
        if stepsize > 0:
            if n % stepsize == 0:
                lr = base_lr * (1 + gamma * float(n)/stepsize)**(- power)
                scale = base_scale * (1 + scale_gamma * float(n)/stepsize)**(- scale_power)
        else:
            step= stepvalue[step_idx]
            if n > step:
                step_idx += 1
            #lr =  base_lr*decay**(step_idx)
            #scale = base_scale*scale_decay**step_idx


        ### minibatch
        n_grp = n % iter_per_epoch
        if  n_grp == 0:
            logging.info("shuffle sample")
            np.random.shuffle(sample)
        mini = sample[minibatch_size*n_grp:minibatch_size*(n_grp+1)]
        mini = np.sort(mini)
        if  not use_tf_seed:
            c_noise =  sp.stats.cauchy.rvs(loc=0, scale=scale, size=num_Cauchy_seeds)
            new_mini = np.zeros((minibatch_size,num_Cauchy_seeds))
            for i in range(minibatch_size):
                for j  in range(num_Cauchy_seeds):
                    new_mini[i][j] = mini[i] - c_noise[j]
            new_mini = new_mini.flatten()
            mini = np.sort(new_mini)

        mini = np.array(mini, dtype=np.complex128)



        ### run

        if use_summary:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            summary, new_loss, new_Psigma, new_PA = sess.run([merged, loss, Psigma, PA],\
                feed_dict={ minibatch_op:mini, scale_op:scale},\
                options=run_options,
                run_metadata=run_metadata)
            summary_writer.add_run_metadata(run_metadata, 'step%03d' % n)
            summary_writer.add_summary(summary, n)

        else:
            new_loss, new_Psigma, new_PA = sess.run([ loss, Psigma, PA],\
                feed_dict={ minibatch_op:mini, scale_op:scale})

        if comparing_with_dirct:
            sc.eps= scale
            direct_deriv, direct_loss = sc.deriv_loss(diag_A, sigma, mini)
            logging.info("loss:separete - direct = {}".format(new_loss - direct_loss) )
            logging.info("PA:separete - direct = \n{}".format(new_PA - direct_deriv) )
            logging.info("PA:separete = \n{}".format(new_PA) )
            logging.info("PA:direct = \n{}".format(direct_deriv) )



        #logging.info("new_Psigma:{}".format(new_Psigma))
        #logging.info("new_PA:\n{}".format(new_PA))
        m_Psigma = lr*new_Psigma + momentum*old_Psigma
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
            if diag_A[k] > max_sq_sample*1.2:
                logging.info( "diag_A[{}]={} reach the boundary".format(k,diag_A[k]))
                diag_A[k] = max_sq_sample - 0.01 + 0.001*np.random.randn(1)

        if update_sigma:
            logging.debug( "m_Psigma=", m_Psigma)
            if n > start_update_sigma:
                sigma -= reg_coef_sigma*m_Psigma
            if abs(sigma**2) > max_sq_sample:
                logging.info("sigma reachs the boundary:{}".format(sigma))

        old_m_PA= np.copy(m_PA)
        old_Psigma = np.copy(m_Psigma)

        ### assign update
        sess.run(tf.assign(diag_A_op,diag_A))
        sess.run(tf.assign(sigma_op,sigma))


        ######################################
        ### Monitoring
        #####################################
        test_loss=np.linalg.norm(np.sort(diag_A) - n_test_diag_A) +  abs(sigma - n_test_sigma)
        average_loss += new_loss
        average_test_loss += test_loss
        average_sigma += sigma
        average_diagA += diag_A
        if n %  log_step == 0:
            if n > 0:
                average_loss /= log_step
                average_test_loss /= log_step
                average_sigma /= log_step
                average_diagA /= log_step
            #average_sigma *= normalize_ratio
            #average_diagA *= normalize_ratio
            logging.info("{}-iter:lr= {}, scale {}".format(n,lr,scale, ))
            logging.info("loss= {}  / average".format( average_loss))
            logging.info("test_loss= {}  / average".format(average_test_loss))
            logging.info("sigma= {}  / average".format(average_sigma))
            #logging.info("diag_A (sorted)=\n{}  / 10-iter".format(np.sort(average_diagA)))
            #logging.info( "diag_A (raw) =\n{}".format(diag_A))
            logging.info("diag_A =\n{}  / average".format(np.sort(average_diagA)))
            #logging.info("diag_A/ average: min, mean, max= \n {}, {}, {} ".format(np.min(average_diagA), np.average(average_diagA), np.max(average_diagA)))

            average_loss = 0
            average_test_loss = 0
            average_sigma = 0
            average_diagA = 0

        logging.debug( "{}-iter:lr={},scale{}, loss: {}".format(n,lr, scale, loss) )
        logging.debug( "sigma: {}".format(sigma))
        #diag_A=np.sort(diag_A)
        logging.debug( "test_loss: {}".format(test_loss))
        logging.debug( "mean_of_diagA: {}, var_of_A={}".format(np.average(diag_A), np.var(diag_A) ))
        logging.debug( "diag_A (raw) =\n{}".format(diag_A))


        if plot_stepsize > 0 and n % plot_stepsize == 0 and n > 0:
            logging.info("Plotting density...")
            plt.clf()
            plt.figure()
            sc = SemiCircular(eps=scale)
            y_axis_truth = sc.square_density(x_axis,np.diag(n_test_diag_A), n_test_sigma)
            plt.plot(x_axis,y_axis_truth, label="Truth")
            #plt.plot(x_axis,y_axis_init, label="Init")
            y_axis = sc.square_density(x_axis,np.diag(diag_A), sigma)
            plt.plot(x_axis,y_axis, label="{}-iter".format(n))
            plt.legend()
            plt.savefig("images/tf_test_optimize/plot_density_{}-iter".format(n))
            plt.clf()
            logging.info("Plotting density...done")
    if use_summary:
        summary_writer.close()
    sigma *= normalize_ratio
    diag_A *= normalize_ratio
    logging.info("result:")
    logging.info("sigma:".format(sigma))
    logging.info("diag_A:\\{}".format(diag_A))
    return diag_A, sigma




def tf_test_optimize():
    dim = i_dim
    num_sample =1



    jobname = "tf_test_optimize"
    #to file
    file_log = logging.FileHandler('log/{}.log'.format(jobname), 'w')
    file_log.setLevel(logging.INFO)
    file_log.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(file_log)
    #logging.getLogger().setLevel(logging.INFO)
    #param = 2 + np.random.rand(size)*3
    #param = np.sort(param)
    #"""
    param = np.ones(dim)*5
    half_dim = int(dim/2)
    #param = 2*np.random_sample(dim)
    for i in range(half_dim):
        param[i] = 2
    #"""
    logging.info( "truth=\n{}".format(np.array(param)))
    param = np.sort(param)
    param_mat = np.diag(param)
    sigma = 1
    evs_list =[]
    for i  in range(num_sample):
        evs= np.linalg.eigh(info_plus_noise(dim, param_mat,sigma, COMPLEX=True))[0]
        evs_list += evs.tolist()
    if num_sample == 1:
        logging.info( "sqrt(sample)=\n{}".format(sp.sqrt(np.array(evs_list))))

    #mean_param = 2
    sq_sample = np.array(sp.sqrt(evs_list))
    mean_param =np.average(sq_sample)

    result, r_v= monitor_optimization(dim, param, sigma, evs_list)
    plt.figure()

    plt.plot(param, label="Truth")
    plt.plot(sq_sample, label="Sample")
    plt.plot(result, label="Result")
    plt.legend()
    plt.savefig("images/tf_test_optimize/{}.png".format(jobname))
    logging.getLogger().removeHandler(file_log)


tf_test_optimize()
"""
def function(n):
    return tf_test_optimize()
def multi(n):
    p = Pool(4)  #maximal number of processes: 7
    result = p.map(function, range(n))
    return result

def main():
    data = multi(4)
    for i in data:
        print(i)
main()
"""
#for random_minibatch in [True, False]:
#    for minibatch in [8,16,32]:
#        jobname = "5-2_32x32_ss_mb-{}_rm-{}".format(minibatch, random_minibatch)
#        test_optimize(size=32, num_sample=1, minibatch=minibatch, random_minibatch=random_minibatch, jobname=jobname)
