import tensorflow as tf
import numpy as np
import scipy as sp
from timer import Timer
from tf_recurrent_cauchy_net import *
from matrix_util import *
from random_matrices import *

from cauchy import SemiCircular

import matplotlib.pyplot as plt


num = 5
scale = 1e-1

seed = Cauchy_noise(num,0,scale)


sess = tf.InteractiveSession()
### Summary
merged = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('./tb_log_rcn', graph=sess.graph)
### Initialize
sess.run(tf.global_variables_initializer())

num_iter = 1000
out = []
for i in range(num_iter):
    noise = sess.run(seed).real
    noise = np.average(noise)
    print(noise)
    out.append(noise)

out = np.array(out).flatten()
print ("average_of_average", np.average(out))

plt.figure()
plt.hist(out,bins=100, normed=True)
this_dir = os.getcwd()
log_dir ="{}/images".format(this_dir)

plt.savefig("{}/tf_cauchy.png".format(log_dir))
plt.clf()
