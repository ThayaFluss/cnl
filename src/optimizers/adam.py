import math
import numpy as np






class Adam(object):
    """ Adam."""
    def __init__(self, alpha=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        super(Adam, self).__init__()
        self.alpha=alpha
        self.beta1 = beta1
        self.beta2=beta2
        self.eps=eps

        self.m = None
        self.v = None
        self.t = 0

        self.MultipleScheduler = None


    def setup(self, param, Scheduler):
        self.MultipleScheduler = Scheduler
        self.m = np.zeros_like(param)
        self.v = np.zeros_like(param)


    def lr(self):
        return self.alpha*self.MultipleScheduler.get(self.t)


    def update(self, param, grad):
        self.t += 1

        m, v = self.m, self.v
        m += (1 - self.beta1) * (grad - m)
        ####
        # m = self.beta1*m + (1- self.beta1)*grad
        v += (1 - self.beta2) * (grad * grad - v)

        vec = m/(np.sqrt(v) + self.eps)

        fix1 = 1. - math.pow(self.beta1, self.t)
        fix2 = 1. - math.pow(self.beta2, self.t)
        coeff = self.lr() * math.sqrt(fix2) / fix1

        param -= coeff*vec
