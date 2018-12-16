import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import logging
import env_logger
from matrix_util import *

class SPNMonitor(object):
    """ Monitor for training spn"""
    def __init__(self, log_step, stdout_step, plot_stepsize=-1, \
    monitor_validation=True,monitor_Z=True, zero_thresholds=[]):
        super(SPNMonitor, self).__init__()
        self.log_step = log_step
        self.stdout_step = stdout_step
        self.plot_stepsize = plot_stepsize
        self.axis = None



        self.test_params = dict()
        self.monitor_validation = monitor_validation

        self.train_loss_list = []
        self.ave_train_loss = 0

        self.val_loss_list = []
        self.ave_val_loss = 0


        self.ave_forward_iter = np.zeros(1)
        self.total_average_forwad_iter = np.zeros(1)

        self.num_zeros = []
        self.zero_thresholds = zero_thresholds

        self.Optimizer = None

        ### for computing Z-value ###
        self.monitor_Z = monitor_Z
        self.test_U = None
        self.test_V = None
        self.sq_sample = None

        ### counter
        self.t = 0

        ###
        self.dirname = "../images/train_rnn_sc"


    def setup(self, diag_A, sigma, Optimizer, Sampler, U=None,V=None):
        if diag_A is None:
            self.monitor_validation = False
        else:
            self.test_params["diag_A"] =  diag_A
            self.test_params["sigma"] = sigma
            self.monitor_validation = True

        self.Optimizer = Optimizer
        self.Sampler = Sampler
        sample = Sampler.sample
        self.x_axis = np.linspace(-max(sample)*4, max(sample)*4, 201) ## Modify


        self.test_U = U
        self.test_V = V

    def val_loss(self, sc):
        assert self.monitor_validation
        test_diag_A = self.test_params["diag_A"]
        test_sigma = self.test_params["sigma"]
        #val_loss=np.sum(np.abs(np.sort(np.abs(diag_A)) - np.sort(np.abs(n_test_diag_A)))) +  np.abs(np.abs(sigma) - n_test_sigma)
        val_loss=np.linalg.norm(np.sort(np.abs(sc.diag_A)) - np.sort(np.abs(test_diag_A))) +  np.abs(np.abs(sc.sigma) - test_sigma)
        return val_loss

    def collect(self, loss, sc):
        ######################################
        ### Monitoring
        #####################################
        ### gathering results ###
        self.train_loss_list.append(loss)
        self.ave_forward_iter[0] += sc.forward_iter[0]
        sc.forward_iter[0] = 0


        diag_A = sc.diag_A
        sigma = sc.sigma


        if self.monitor_validation:
            val_loss = self.val_loss(sc)
            self.val_loss_list.append(val_loss)
            self.ave_val_loss += val_loss




        self.num_zeros.append( \
        [  np.where( np.abs(diag_A) < self.zero_thresholds[l])[0].size \
        for l in range(len(self.zero_thresholds))])

        self.ave_train_loss += loss

        log_step = self.log_step
        n = self.t
        if (n % log_step  + 1 )== log_step:
            ### Count zeros under several thresholds
            num_zero = self.num_zeros[-1]

            if n > 0:
                self.ave_train_loss /= log_step
                self.ave_forward_iter /= log_step
                self.total_average_forwad_iter += self.ave_forward_iter
                if self.monitor_validation:
                    self.ave_val_loss /= log_step
                if (n % self.stdout_step + 1) == self.stdout_step:
                    lr = self.Optimizer.lr()
                    scale = sc.scale
                    #logging.info("{}/{}-iter:".format(n+1,max_iter))
                    logging.info("lr = {0:4.3e}, scale = {1:4.3e}".format(lr,scale) )
                    logging.info("train loss= {}".format( self.ave_train_loss))
                    if self.monitor_Z:
                        p_dim = self.test_U.shape[1]
                        dim = self.test_V.shape[0]
                        sq_sample = sp.sqrt(self.Sampler.sample)[::-1]
                        diff_sv =  sq_sample - np.sort(abs(diag_A))[::-1]
                        Diff = self.test_U @ rectangular_diag( diff_sv, p_dim, dim) @ self.test_V
                        z_value = np.sum(Diff)/ (sp.sqrt(p_dim)*sigma)
                        logging.info("z value = {}".format(z_value))


                if self.monitor_validation:

                    if (n % self.stdout_step + 1) == self.stdout_step:
                        logging.info("val_loss= {}".format(self.ave_val_loss))
                if (n % self.stdout_step + 1) == self.stdout_step:

                    logging.info("sigma= {}".format(sigma))
                    #logging.info("diag_A (sorted)=\n{}  / 10-iter".format(np.sort(diag_A)))
                    #logging.info( "diag_A (raw) =\n{}".format(diag_A))
                    logging.info("num_zero={} / thres={}".format(num_zero,self.zero_thresholds))
                    #logging.info("diag_A/ average: min, mean, max= \n {}, {}, {} ".format(np.min(average_diag_A), np.average(average_diag_A), np.max(average_diag_A)))
                    logging.info("forward_iter={}".format(self.ave_forward_iter))


                self.ave_train_loss = 0
                self.ave_forward_iter[0] = 0
                if self.monitor_validation:
                    self.ave_val_loss = 0


        if self.plot_stepsize > 0 and n % self.plot_stepsize == 0:
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
            plt.plot(x_axis,y_axis, label="{}-iter".format(n+1), color="blue")
            plt.legend()
            plt.savefig("{}/plot_density_{}-iter".format(dirname, n+1),dpi=i_dpi)
            plt.clf()
            plt.close()
            logging.info("Plotting density...done")

        self.t += 1
