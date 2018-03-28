import numpy as np
from numba import jit, f8, i4
from timer import Timer

@jit('f8[:,:](i4,f8[:,:],f8[:,:],i4)')
def many_matmul(num, A,B, dim):
    C = np.eye(dim,dtype=np.float64)
    for i in range(num):
        C = A @ C @ B
    return C



@jit('i4(i4,i4)')
def test_matmul(num,dim):
    A = np.random.randn(dim,dim)
    B = np.random.randn(dim,dim)
    C = many_matmul(num, A, B, dim)
    return 0



def test_matmul_row(num, dim):
    A = np.random.randn(dim,dim)
    B = np.random.randn(dim,dim)
    C = np.eye(dim,dtype=np.float64)
    for i in range(num):
        C = A @ C @ B
    return 0


num = 300
dim = 100
test_matmul(num,dim)

timer = Timer()
timer.tic()
test_matmul(num,dim)

timer.toc()
print (timer.total_time)

timer = Timer()
timer.tic()
test_matmul_row(num,dim)
timer.toc()

print (timer.total_time)
