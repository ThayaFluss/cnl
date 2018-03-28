import tensorflow as tf
import numpy as np
from timer import Timer


def L1_tester(X,Y):
    L1_norm= tf.reduce_mean(tf.abs(X-Y))
    print("L1_norm=",sess.run(L1_norm))


a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = a @ b

d = tf.constant(np.arange(4).reshape([2,2]))
diag = tf.diag_part(d)
out = diag[::-1]

d= 3
ones = tf.ones(d, tf.complex128)
entries = [
            [1*ones,2*ones],\
            [3*ones,4*ones]\
            ]
Enlarge = tf.matrix_diag(entries)

sigma = tf.Variable(1.+2.*1j, tf.complex128)

###matrix product with vector
A = tf.constant(np.arange(16), shape=[4,4], dtype=tf.int32)
v = tf.constant(np.arange(4),shape = [4,1], dtype=tf.int32)
B = A@v

# Creates a session with log_device_placement set to True.
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
# Runs the op.
#print(sess.run(sigma**2*Enlarge))


sess = tf.Session()
sess.run(tf.global_variables_initializer())
num_test = 2000





print("###convert vs concat")
###win : convert
a = tf.constant([1.0,2])
b = tf.constant([1.0,3])
c = tf.constant([1.0,4])
d = tf.constant([1.0,5])

out_ctt = tf.convert_to_tensor([a,b,c,d])
out_concat = tf.concat([a,b,c,d],0)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

timer = Timer()
timer.tic()
for n in range(num_test):
    sess.run(out_ctt)

timer.toc()
print ("ctt:{}".format(timer.total_time))

timer = Timer()
timer.tic()
for n in range(num_test):
    sess.run(out_concat)
timer.toc()
print ("concat\t:{}".format(timer.total_time))



print("###test partial batch matmul \n for_roop vs tiling")
### winer : tiling
G = np.arange(2**2, dtype=np.float).reshape([2,2])
G = tf.Variable(G, tf.float64)
E = np.zeros([2,2,2,2])
for i in range(2):
    for j in range(2):
        E[i][j][i][j] = 1.
E = tf.convert_to_tensor( E)
# classical
out = []
for i in range(2):
    for j in range(2):
        entry = G @ E[i][j] @ G
        out.append(entry)
out_ctt = tf.convert_to_tensor(out)
out_ctt = tf.reshape(out_ctt, [2,2,2,2])
# new
G = tf.reshape(G, [4])
G_e = tf.tile(G, [4])
G_e = tf.reshape(G_e, [2,2,2,2])
out_tile = G_e @ E @ G_e
sub = out_tile - out_ctt

sess = tf.Session()
sess.run(tf.global_variables_initializer())

timer = Timer()
timer.tic()
for n in range(num_test):
    sess.run(out_ctt)

timer.toc()
print ("ctt:{}".format(timer.total_time))

timer = Timer()
timer.tic()
for n in range(num_test):
    sess.run(out_tile)
timer.toc()
print ("tile:{}".format(timer.total_time))

sub = sess.run(sub)
print("sub=", sub)





print("###test partial batch matmul (large)\n for_roop vs tiling")
### winer : tiling
G = np.arange(2**2, dtype=np.float).reshape([2,2])
G = tf.Variable(G, tf.float64)
dim = 60
E = np.arange(dim*2**2, dtype=np.float).reshape([dim, 2,2])
E = tf.convert_to_tensor( E)
# classical
out = []
for i in range(dim):
        entry = G @ E[i]@ G
        out.append(entry)
out_ctt = tf.convert_to_tensor(out)
out_ctt = tf.reshape(out_ctt, [dim,2,2])
# new
G = tf.reshape(G, [4])
G_e = tf.tile(G, [dim])
G_e = tf.reshape(G_e, [dim,2,2])
out_tile = G_e @ E @ G_e
sub = out_tile - out_ctt

sess = tf.Session()
sess.run(tf.global_variables_initializer())

timer = Timer()
timer.tic()
for n in range(num_test):
    sess.run(out_ctt)

timer.toc()
print ("ctt:{}".format(timer.total_time))

timer = Timer()
timer.tic()
for n in range(num_test):
    sess.run(out_tile)
timer.toc()
print ("tile:{}".format(timer.total_time))

sub = sess.run(sub)
print("sub=", sub)


print("###test_matrix_of_vector")
### roop vs flatten : draw
J_X = np.arange(2**4, dtype=np.float64).reshape([2,2,2,2]) + 1.
length = 60
vec = np.arange(length, dtype=np.float64) + 1.
J_X = tf.constant(J_X, tf.float64)
vec = tf.constant(vec, tf.float64)

### new

out_flatten = tf.reshape(J_X, [2**4] )* tf.reduce_mean(vec)

### classical roop
out_roop = []
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                ### d
                temp = J_X[i][j][k][l]*vec
                out = tf.reduce_mean(temp)
                out_roop.append(out)
out_roop = tf.convert_to_tensor(out_roop)


sub = out_roop - out_flatten

sess = tf.Session()
sess.run(tf.global_variables_initializer())

timer = Timer()
timer.tic()
for n in range(num_test):
    sess.run(out_roop)

timer.toc()
print ("roop:{}".format(timer.total_time))

timer = Timer()
timer.tic()
for n in range(num_test):
    sess.run(out_flatten)
timer.toc()
print ("flatten:{}".format(timer.total_time))

sub = sess.run(sub)
print("sub=", sub)



print ("test_broad_cast")
a = np.arange(2*6).reshape([2,6])
vec = np.arange(6)
a = tf.Variable(a)
vec = tf.Variable(vec)
x = a*vec

sess = tf.Session()
sess.run(tf.global_variables_initializer())
x = sess.run(x)
print (x)



print("test_tensordot")
dim = 30

X = np.random.randn(2*2*dim)
J_X = np.random.randn(2**4)
J_det = np.random.randn(2*2*dim)
det = np.arange(dim)

X = tf.constant(X, shape=[2,2,dim],dtype=tf.float64)
J_X = tf.constant(J_X, shape=[2,2,2,2],dtype=tf.float64)
J_det = tf.constant(J_det, shape=[2,2,dim],dtype=tf.float64)
det = tf.constant(det, shape=[dim],dtype=tf.float64)+1

D = []
for i in range(2):
    for j in range(2):
        ### entry of matrix
        for k in range(2):
            for l in range(2):
                temp2=   X[k][l]*J_det[i][j]/(det**2)
                temp2 = tf.reduce_mean(temp2)
                D.append(temp2)
out = tf.convert_to_tensor(D)
out = tf.reshape(out, [2,2,2,2])

entry = tf.tensordot( J_det,X/(det**2), [[2],[2]])
entry /= dim
out_new = entry
#out_new = tf.reshape(entry, [2,2,2,2])

L1_tester(out, out_new)
sub = out - out_new


sess = tf.Session()
sess.run(tf.global_variables_initializer())

timer = Timer()
timer.tic()
for n in range(num_test):
    sess.run(out)

timer.toc()
print ("roop:{}".format(timer.total_time))

timer = Timer()
timer.tic()
for n in range(num_test):
    sess.run(out_new)
timer.toc()
print ("flatten:{}".format(timer.total_time))

sub = sess.run(sub)
print("sub=", sub)
