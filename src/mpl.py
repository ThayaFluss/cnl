import scipy as sp
import numpy as np

import scipy.stats as ss
import matplotlib.pyplot as plt
from random_matrices import *
"""
Validate Liklihood Equation for Marchenko-Pastur Law

"""
from tqdm import tqdm, trange

def main():
    plot_log_value()
    #plot_loss()



def plot_loss():
    v0 = 0.5 #1/sp.sqrt(2)
    d = 100
    num_cauchy_rv = 100
    num_test = 1
    max_x = 1
    num_x = max_x*10*d
    num_x = int(num_x)
    #num_x = 1.2*d*num_cauchy_rv ## for plot
    x = np.linspace(0.01,1, num_x)





    x = np.linspace(0.01,1, num_x)

    plt.figure(figsize=(3.14*2,3.14*2))  ## 3.14: 8cm
    plt.rcParams['font.family'] ='sans-serif'
    plt.rcParams['font.size'] = 8
    #lines = ["-", "--", "-.", ":"]
    lines = [":", "-.", "--", "-"]
    #plt.rc("text", usetex=True)
    plt.title("Plot of Empirical Loss")
    plt.xlabel("v")
    plt.xticks([0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.ylabel("Empirical Loss")
    idx = 0



    eigs = sample_eigen_values(v0,d)
    for scale in [  1, 1e-1, 1e-2, 1e-3]:
        print("scale=", scale)
        log_vals =  np.zeros_like(x)
        perturbed = perturbation(eigs, d,scale, num_cauchy_rv)
        result = empirical_loss_multi(x, perturbed, scale)

        plt.plot(x,result , linestyle=lines[idx], label="$\gamma=${}".format(scale))
        idx += 1
        idx %= len(lines)

    #plt.yscale("log")
    plt.grid(which='major',color='black',linestyle='-')
    plt.legend()
    plt.show()




def plot_log_value():
    v0 = 0.5 #1/sp.sqrt(2)  ###  parameter for genrating sample
    d = 100 ### dimension
    num_cauchy_rv = 100
    #num_test = 1
    max_x = 1
    num_x = max_x*10*d
    num_x = int(num_x)
    x = np.linspace(0.01,1, num_x)

    fig= plt.figure(figsize=(3.14*2,3.14*2))  ## 3.14: 8cm
    plt.rcParams['font.family'] ='sans-serif'
    plt.rcParams['font.size'] = 10

    #lines = ["-", "--", "-.", ":"]
    lines = [":", "-.", "--", "-"]
    #plt.rc("text", usetex=True)
    x_list = [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    ax_h = fig.add_subplot(2,1,1, \
    title="Log-scale plot of $h$",\
    yscale="log",\
    xlabel="v",\
    xticks= x_list,\
    ylabel="$h$")

    ax_loss = fig.add_subplot(2,1,2, \
    title="Mean Cauchy Noise Loss",\
    xlabel="v",\
    xticks= x_list,\
    ylabel="Loss")


    idx = 0
    eigs = sample_eigen_values(v0,d)
    for scale in [ 1, 1e-1, 1e-2, 1e-3]:
        print("scale=", scale)
        result =  np.zeros_like(x)
        perturbed = perturbation(eigs, d,scale, num_cauchy_rv)
        loss = empirical_loss_multi(x, perturbed, scale)
        vals = val_mpl_multi(x, perturbed, scale)
        ax_h.plot(x, vals, linestyle=lines[idx], label="$\gamma=${}".format(scale))
        ax_loss.plot(x, loss, linestyle=lines[idx], label="$\gamma=${}".format(scale))
        idx += 1
        idx %= len(lines)



    for ax in [ax_h, ax_loss]:
        ax.grid(which='major',color='gray',linestyle='-')
        ax.legend()

    fig.tight_layout()


    plt.savefig('../../phd/mpl_likelihood_eq_loss.png', transparent=True, dpi=300)
    #plt.show()


def sample_eigen_values(v0,d):
    Z = Ginibre(d,d)  ## mean = 0, variance = 1/d.
    sample_mat = v0*Z.T@Z

    #s1 = s_MLE(sample_mat)
    sample, _ = np.linalg.eigh(sample_mat)

    return sample


def perturbation(eigs, d, scale, num_cauchy_rv):
    perturbed = np.zeros(d*num_cauchy_rv)
    idx = 0
    for k in range(d):
        c_noise =  ss.cauchy.rvs(loc=0, scale=scale, size=num_cauchy_rv)
        for l in range(num_cauchy_rv):
            perturbed[idx] = eigs[k] + c_noise[l]
            idx+=1
    #plt.hist(perturbed, bins=100)
    #plt.savefig("../images/mpl_perturbed.png")
    #plt.clf()
    #plt.close()
    return  perturbed

"""
Estimation of sigma by MLE
"""
def s_MLE(sample_mat):
    r = np.mean( sample_mat**2)
    r *= sample_mat.shape[1]
    r = sp.sqrt(r)

    return r

def val_mpl_multi(x,sample,scale):
    n = np.size(x)
    v = np.zeros(n)
    for k in range(n):
        v[k] = val_mpl(x[k], sample, scale)

    return v

def val_mpl(v,  sample, scale):
    assert scale > 0
    z = sample +  1j*scale
    ratio = np.abs( z/ (z-4*v))
    value = np.mean(ratio)
    return value

def param_derivative_multi(x, sample,scale):
    n = np.size(x)
    out = np.zeros(n)
    for k in range(n):
        out[k] = param_derivative(x[k], sample,scale)
    return out


def param_derivative(v,sample, scale):
    assert v > 0
    deriv =    val_mpl(v,sample,scale) - 1
    deriv /=   2*v

    return deriv



def empirical_loss(v, sample, scale):
    z = sample + 1j*scale
    f = sp.sqrt( 1/v - 4/z)
    prob = f.imag/sp.pi
    assert np.all( prob> 0)
    out = -np.mean(sp.log(prob))
    return out


def empirical_loss_multi(x, sample,scale):
    n = np.size(x)
    out = np.zeros(n)
    for k in range(n):
        out[k] = empirical_loss(x[k], sample,scale)
    return out



if __name__ == "__main__":
    main()
