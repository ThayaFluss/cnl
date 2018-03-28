import numpy as np
from timer import Timer


### Matrix product with diagonal matrix
def test_matrix_diag_product():
    size = 100
    mat = np.random.rand(size,size)
    mmat = np.matrix(mat)
    d = np.random.rand(size)
    num = 100


    print ( np.allclose(d*mat , mat @ np.diag(d) )) #true
    print ( np.allclose(d*mmat , mmat @ np.diag(d) )) #false
    print ( np.allclose(d.T*mmat , mmat @ np.diag(d) )) #flase
    timer = Timer()
    timer.tic()
    temp = np.zeros((size, size))
    for i in range(num):
        temp += mat + (d*mat) @ mat

    timer.toc()
    print (timer.total_time)
    timer = Timer()
    timer.tic()
    for i in range(num):
        temp += mat + mat @ np.diag(d) @ mat

    timer.toc()
    print (timer.total_time)


class test_class(object):
    """docstring for test_class."""
    def __init__(self):
        super(test_class, self).__init__()
        self.mat = 0


def test_allclose():
    size = 400
    mat = np.random.randn(size,size) + 1j*np.random.randn(size,size)
    thres = 1e-8
    mat2 = mat + thres
    num = 100


    sc = test_class()
    sc.mat = mat

    timer = Timer()
    timer.tic()
    for i in range(num):
        sc.mat = mat
        mat2 = sc.mat
    timer.toc()
    print (timer.total_time)



    timer = Timer()
    timer.tic()
    for i in range(num):
        temp = mat @ mat2
    timer.toc()
    print (timer.total_time)




    timer = Timer()
    timer.tic()
    for i in range(num):
        np.linalg.norm(mat - mat2)
    timer.toc()
    print (timer.total_time)






    timer = Timer()
    timer.tic()
    for i in range(num):
        mat2 = np.copy(mat)
    timer.toc()
    print (timer.total_time)


    timer = Timer()
    timer.tic()
    for i in range(num):
        np.max(abs(mat - mat2))
    timer.toc()
    print (timer.total_time)

    timer = Timer()
    timer.tic()
    for i in range(num):
        np.allclose(mat, mat2)

    timer.toc()
    print (timer.total_time)
    timer = Timer()

    timer.tic()
    for i in range(num):
        np.isclose(mat, mat2)

    timer.toc()
    print (timer.total_time)

def test_while():
    max_iter = 10000
    A = np.eye(64)
    thres = 1000*np.linalg.norm(A)

    timer = Timer()
    timer.tic()
    for i in range(max_iter):
        A += A
        if np.linalg.norm(A) > thres:
            break
    timer.toc()
    print(timer.total_time)

    A = np.eye(64)
    
    timer = Timer()
    timer.tic()
    i=0
    while (np.linalg.norm(A)  < thres):
        A += A
        i+=1
    timer.toc()
    print(timer.total_time)


#test_matrix_diag_product()
#test_allclose()
test_while()
