import math


class Fix(object):
    """ Multiple Scheduler for learning rate """
    """ Constant"""
    def __init__(self):
        super(Fix, self).__init__()

    def get(self,t):
        return 1



class Step(object):
    """ Step."""
    def __init__(self, stepsize, decay=0.5):
        super(Step, self).__init__()
        self.stepsize = stepsize
        self.decay = decay

    def get(self, t):
        mult =  math.pow(t // self.stepsize, self.decay)
        return mult



class Inv(object):
    """ Inv."""
    def __init__(self, gamma=1e-4, power=0.75):
        super(Inv, self).__init__()
        self.gamma = gamma
        self.power = power

    def get(self,t):
        mult = math.pow( 1 + self.gamma*t, -self.power)
        return
