
class Trainer(object):
    """docstring for Trainer."""
    def __init__(self, shape):
        super(Trainer, self).__init__()
        self.shape = shape ## p_dim, dim
        self.base_scale = 1e-1
        self.base_lr = 1e-4
        self.max_epoch = 400
        self.dim_cauchy = 1
        self.minibatch_size = 1
        self.lr_policy = "fix"
        self.step_epochs = []
        self.reg_coef = 0
        self.edge = 1.
        self.sample  = None
        self.normalize_sample = False
        self.clip_grad = -1

        self.monitor = None
        self.optimizer = "Adam"
        self.REG_TYPE = "L1"

        ### for momentum
        self.momentum = 0.9
        self.lr_mult_sigma = 1

        self.alpha = self.base_lr
        self.beta1= 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

        ### inv
        self.gamma = 1e-4
        self.powe = 0.75

        ### step
        self.decay = 0.1
        self.stepvalue = None



        ### monitor
        self.log_step = 50
        self.stdout_step = 1000
        self.KL_log_step = 10*dim
        self.plot_stepsize = -1
        self.stop_count_thres = 100



        ### local variables
        step_idx = 0
        average_loss = 0
        average_val_loss = 0
        average_param = None

        average_forward_iter = np.zeros(1)
        total_average_forwad_iter = np.zeros(1)

    def optimize():


    def steup(self):
        sample_size =len(sample)
        assert sample_size == sample.shape[0]
        iter_per_epoch = int(sample_size/minibatch_size)
        max_iter = max_epoch*iter_per_epoch


        if lr_policy == "step":
            if len(step_epochs) == 0:
                self.stepvalue = [2*iter_per_epoch, max_iter]
            else
                self.stepvalue = iter_per_epoch*np.asarray(step_epochs)


        ### inputs to be denoised
        if self.normalize_sample:
            max_row_sample = max(sample)
            normalize_ratio = sp.sqrt(max_row_sample)
            sample /= max_row_sample
        else:
            normalize_ratio = 1.
