import matplotlib.pyplot as plt




class SPNMonitor(object):
    """ Monitor for training spn"""
    def __init__(self, log_step, stdout_step, plot_stepsize=-1, \
    monitor_validation=True,monitor_Z=True, zero_thresholds=[]):
        super(MonitorSPN, self).__init__()
        self.log_step = log_step
        self.stdout_step = stdout_step
        self.plot_stepsize = plot_stepsize
        self.axis = None




        self.test_params = dict()
        self.monitor_validation = monitor_validation

        self.train_loss = []
        self.val_loss = []

        self.ave_train_loss = 0
        self.ave_val_loss = 0
        self.ave_forward_iter = np.zeros(1)




        self.num_zeros = None
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
        self.dirname = dirname


    def setup(self, diag_A, sigma, Optimizer, Sampler, U=None,V=None):
        self.monitor_validation = True
        self.test_params["diag_A"] =  diag_A
        self.test_params["sigma"] = sigma

        self.Optimizer = Optimizer
        self.Sampler = Sampler
        sample = Sampler.sample
        self.x_axis = np.linspace(-max(sample)*4, max(sample)*4, 201) ## Modify


        self.test_U = U
        self.test_V = V

    def val_loss(self):
        assert self.monitor_validation
        test_diag_A = self.test_params["diag_A"]
        test_sigma = self.test_params["sigma"]
        #val_loss=np.sum(np.abs(np.sort(np.abs(diag_A)) - np.sort(np.abs(n_test_diag_A)))) +  np.abs(np.abs(sigma) - n_test_sigma)
        val_loss=np.linalg.norm(np.sort(np.abs(diag_A)) - np.sort(np.abs(test_diag_A))) +  np.abs(np.abs(sigma) - test_sigma)
        return val_loss

    def collect(self, loss, sc):
        ######################################
        ### Monitoring
        #####################################
        ### gathering results ###
        self.train_loss.append(new_loss)
        self.ave_forward_iter[0] += sc.forward_iter[0]
        sc.forward_iter[0] = 0

        diag_A = sc.diag_A
        sigma = sc.sigma


        if self.monitor_validation:
            val_loss = self.val_loss()
            val_loss_list.append(val_loss)
            self.ave_val_loss += val_loss




        self.num_zeros.append( \
        [  np.where( np.abs(diag_A) < self.zerothresholds[l])[0].size \
        for l in range(len(self.zerothresholds))])

        self.ave_train_loss += new_loss

        log_step = self.log_step
        n = self.t
        if (n % log_step  + 1 )== log_step:
            ### Count zeros under several thresholds
            num_zero = self.num_zeros[-1]

            if n > 0:
                self.ave_train_loss /= log_step
                self.ave_forward_iter /= log_step
                total_average_forwad_iter += self.ave_forward_iter
                if self.monitor_validation:
                    self.ave_val_loss /= log_step
                if (n % stdout_step + 1) == stdout_step:
                    lr = Optimizer.lr()
                    scale = sc.scale
                    #logging.info("{}/{}-iter:".format(n+1,max_iter))
                    logging.info("lr = {0:4.3e}, scale = {1:4.3e}".format(lr,scale) )
                    logging.info("train loss= {}".format( self.ave_train_loss))
                    if self.monitor_Z:
                        p_dim = test_U.shape[1]
                        dim = test_V.shape[0]
                        diff_sv =  np.sort(sq_sample)[::-1] - np.sort(abs(diag_A))[::-1]
                        Diff = test_U @ rectangular_diag( diff_sv, p_dim, dim) @ test_V
                        z_value = np.sum(Diff)/ (sp.sqrt(p_dim)*sigma)
                        logging.info("z value = {}".format(z_value))


                if monitor_validation:

                    if (n % stdout_step + 1) == stdout_step:
                        logging.info("val_loss= {}".format(self.ave_val_loss))
                if (n % stdout_step + 1) == stdout_step:

                    logging.info("sigma= {}".format(sigma))
                    #logging.info("diag_A (sorted)=\n{}  / 10-iter".format(np.sort(diagA)))
                    #logging.info( "diag_A (raw) =\n{}".format(diag_A))
                    logging.info("num_zero={} / thres={}".format(num_zero,self.zero_thresholds))
                    #logging.info("diag_A/ average: min, mean, max= \n {}, {}, {} ".format(np.min(average_diagA), np.average(average_diagA), np.max(average_diagA)))
                    logging.info("forward_iter={}".format(self.ave_forward_iter))


                self.ave_train_loss = 0
                self.ave_forward_iter[0] = 0
                if monitor_validation:
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
