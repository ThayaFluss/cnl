


class Momentum(object):
    """ Momentum SGD """
    def __init__(self, base_lr, momentum=0.9, Scheduler=None):
        super(Momentum, self).__init__()
        self.momentum = momentum
        self.base_lr= base_lr
        self.g = None
        self.MultipleScheduler = None
        self.t = 0

    def setup(self, param, Scheduler):
        self.MultipleScheduler = Scheduler
        self.g = np.zeros_like(param)


    def lr(self):
        return self.base_lr*self.MultipleScheduler.get(self.t)


    def update(self, param, grad):
        self.g = self.lr*grad + self.momentum*self.g
        param -= self.g
        self.t += 1
