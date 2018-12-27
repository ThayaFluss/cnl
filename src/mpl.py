import scipy as sp
import numpy as np

import scipy.stats as ss
import matplotlib.pyplot as plt
from random_matrices import *
"""
Validate Liklihood Equation for Marchenko-Pastur Law

"""

def main():
    plot(1, 100, 1e-5, 200)


def plot(s0,  d, scale, num_cauchy_rv):
    Z = Ginibre(d,d)  ## mean = 0, variance = 1/d.
    sample_mat = s0**2*Z.T@Z

    #s1 = s_MLE(sample_mat)

    sample, _ = np.linalg.eigh(sample_mat)

    perturbed = np.zeros(d*num_cauchy_rv)
    idx = 0
    for k in range(d):
        c_noise =  ss.cauchy.rvs(loc=0, scale=scale, size=num_cauchy_rv)
        for l in range(num_cauchy_rv):
            perturbed[idx] = sample[k] + c_noise[l]
            idx+=1

    plt.hist(perturbed, bins=100)
    plt.savefig("../images/mpl_perturbed.png")
    plt.clf()
    plt.close()
    #v0 = val_mpl(s0, perturbed, scale)
    #v1 = val_mpl(s1, perturbed, scale)
    num = 120
    x = np.linspace(0,1.2, num)

    vals = np.zeros(num)
    log_vals = np.zeros(num)
    #plt.ylim(0,2)
    for k in range(num):
        v = val_mpl(x[k], perturbed, scale)
        log_vals[k] = sp.log(v)
        vals[k] = v

    plt.plot(x,vals)
    plt.plot(x,np.ones(num), color="black")
    #plt.yscale("log")
    plt.title("mpl-function: scale={}".format(scale))
    plt.savefig("../images/mlp_vals.png")
    plt.clf()
    plt.close()

    ### log
    plt.plot(x,log_vals)
    plt.yscale("log")
    plt.title("log of mpl-function: scale={}".format(scale))
    plt.savefig("../images/mlp_log_vals.png")
    plt.show()
    import pdb; pdb.set_trace()

    return  v0

"""
Estimation of sigma by MLE
"""
def s_MLE(sample_mat):
    r = np.mean( sample_mat**2)
    r *= sample_mat.shape[1]
    r = sp.sqrt(r)

    return r



def val_mpl(s,  sample, scale):
    n = np.size(sample)
    assert n > 0
    sum = 0
    for k in range(n):
        sum +=  _ratio( sample[k]+ 1j*scale, s**2)
    sum /= n
    return sum


def _ratio(z, v):
    return  abs( z/ (z - 4*v) )



if __name__ == "__main__":
    main()
