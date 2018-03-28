import tensorflow as tf
import numpy as np
from timer import Timer
from tf_recurrent_cauchy_net import *



A = tf.Variable(2)
B = tf.Variable(3)


def graph():
    graph = A
    for d in range(5):
        graph = graph + B*graph
        tf.assgin(A,graph)
    return graph

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('./tb_log_test_roop', graph=sess.graph)
    sess.run(tf.global_variables_initializer())
    out = sess.run(graph())
    print(out)
