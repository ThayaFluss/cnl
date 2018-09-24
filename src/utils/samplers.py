import scipy as sp
import numpy as np


class CauchyNoiseSampler(object):
    """docstring for CauchyNoiseSampler."""
    def __init__(self, scale, minibatch_size=1,  num_cauchy_rv=1,\
     sampling_type="CHOICE", mix="DIAGONAL"):
        super(CauchyNoiseSampler, self).__init__()
        self.scale=scale
        self.minibatch_size = minibatch_size
        self.num_cauchy_rv = num_cauchy_rv
        self.sampling_type = sampling_type
        self.mix = mix
        self.t = 0

        self.sample = None

    def setup(self, sample):
        self.sample = np.copy(sample)


    def get(self):
        """
        Choose minibatch from sample
        iter_per_epoch = len(sample)/minibatch_size
        """
        scale=self.scale
        minibatch_size=self.minibatch_size
        num_cauchy_rv = self.num_cauchy_rv
        SAMPLING = self.sampling_type
        MIX =self.mix
        n = self.t

        sample = self.sample


        if SAMPLING == "SHUFFLE":
            mb_idx = n % minibatch_size
            if mb_idx == 0:
                np.random.shuffle(sample)
            mini = sample[minibatch_size*mb_idx:minibatch_size*(mb_idx+1)]
        elif SAMPLING == "CHOICE":

            mini = np.random.choice(sample, minibatch_size)

        else:
            raise ValueError("SAMPLING is SHUFFLE or CHOICE")

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
