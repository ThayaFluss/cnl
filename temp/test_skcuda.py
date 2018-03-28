import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.linalg as linalg
import skcuda.misc as misc
import skcuda.cublas as cublas

from timer import Timer


def test_matrix_product(num=10, size1=100, size2=100, size3=100):
    lda = size1
    ldb = size2
    ldc = size1


    a = np.asarray(np.random.rand(size1, size2), np.complex128)
    b = np.asarray(np.random.rand(size2, size3), np.complex128)
    c  = np.asarray(np.empty((size1,size3)),np.complex128)


    ###CPU
    timer = Timer()
    timer.tic()
    for i in range(num):
        c_cpu = a @ b
    timer.toc()
    print ("numpy:", timer.total_time)
    #print ("c_cpu=\n",c_cpu)

    ###GPU + skcuda.cublas
    ###Pay attention to transpose
    h = cublas.cublasCreate()
    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    c_gpu = gpuarray.to_gpu(c)
    timer = Timer()
    timer.tic()
    ### Zgemm : for complex128
    ### Sgemm : for float32
    for i in range(num):
        cublas.cublasZgemm(handle=h, transa='t', transb='t', m=size1, n=size3, k=size2,\
        alpha=1, A= a_gpu.gpudata, lda=lda, B=b_gpu.gpudata, ldb=ldb, beta=0, C=c_gpu.gpudata, ldc=ldc)
    timer.toc()

    c_gpu_out = c_gpu.get()

    print ("skcuda.cublas:", timer.total_time)
    #print ("c_gpu=\n", c_gpu_out)
    ### TODO Trsnspose  is too slow
    print (np.allclose(c_cpu.T,c_gpu_out ) )
    cublas.cublasDestroy(h)

    ###GPU + skcuda.linalg
    linalg.init()
    h = cublas.cublasCreate()

    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    c_gpu = gpuarray.to_gpu(c)
    timer = Timer()
    timer.tic()
    for i in range(num):
        c_gpu = linalg.dot(a_gpu, b_gpu)
    timer.toc()
    print ("skcuda.linalg:",timer.total_time)
    print (np.allclose(c_cpu,c_gpu.get() ) )
    cublas.cublasDestroy(h)

    ###GPU + skcuda.linalg
    linalg.init()
    h = cublas.cublasCreate()

    a_gpu_0 = gpuarray.to_gpu(a)
    a_gpu_1 = gpuarray.to_gpu(a)
    a_gpu_2 = gpuarray.to_gpu(a)
    a_gpu_3 = gpuarray.to_gpu(a)
    a_gpu_4 = gpuarray.to_gpu(a)
    a_gpu_5 = gpuarray.to_gpu(a)
    a_gpu_6 = gpuarray.to_gpu(a)
    a_gpu_7 = gpuarray.to_gpu(a)
    a_gpu_8 = gpuarray.to_gpu(a)
    a_gpu_9 = gpuarray.to_gpu(a)

    timer = Timer()
    timer.tic()

    b_gpu = gpuarray.to_gpu(b)
    c_gpu = gpuarray.to_gpu(c)
    c_gpu = linalg.dot(a_gpu_0, b_gpu)
    c_gpu = linalg.dot(a_gpu_1, b_gpu)
    c_gpu = linalg.dot(a_gpu_2, b_gpu)
    c_gpu = linalg.dot(a_gpu_3, b_gpu)
    c_gpu = linalg.dot(a_gpu_4, b_gpu)
    c_gpu = linalg.dot(a_gpu_5, b_gpu)
    c_gpu = linalg.dot(a_gpu_6, b_gpu)
    c_gpu = linalg.dot(a_gpu_7, b_gpu)
    c_gpu = linalg.dot(a_gpu_8, b_gpu)
    c_gpu = linalg.dot(a_gpu_9, b_gpu)

    timer.toc()
    print ("skcuda.linalg(multi):",timer.total_time)
    print (np.allclose(c_cpu,c_gpu.get() ) )
    cublas.cublasDestroy(h)





def test_inverse(num=100, size = 100):

    ### TODO np.float32 has error for inverse by gpu
    a = np.asarray(np.random.rand(size, size), np.complex128)
    ###CPU
    timer = Timer()
    timer.tic()
    for i in range(num):
        a_inv_cpu = np.linalg.inv(a)
    timer.toc()
    print ("numpy:", timer.total_time)

    """
    ###GPU + skcuda.cublas
    ###Pay attention to transpose
    a = a.reshape(1,size,size)
    h = cublas.cublasCreate()
    batchSize = 1
    p = np.asarray(np.empty((size,batchSize)), np.int)
    info = np.asarray(np.empty(batchSize), np.int)
    timer = Timer()
    timer.tic()
    a_gpu = gpuarray.to_gpu(a)
    p_gpu = gpuarray.to_gpu(p)
    info_gpu = gpuarray.to_gpu(info)
    cublas.cublasSgetrfBatched(handle=h, n=size, A=a_gpu.gpudata, lda=size,P=p_gpu.gpudata, info=info_gpu.gpudata, batchSize=1)
    """


    linalg.init()
    a_gpu = gpuarray.to_gpu(a.T)
    timer = Timer()
    timer.tic()
    for i in range(num):
        a_inv_gpu = linalg.inv(a_gpu, lib='cusolver')

    a_inv_gpu_out = a_inv_gpu.get()
    timer.toc()
    print ("skcuda.linalg:",timer.total_time)
    print (np.allclose(a @ a_inv_gpu_out, np.eye(size) ) )


test_matrix_product()
