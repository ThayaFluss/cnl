# -*-encode: utf-8-*-

import time
import numpy as np
import cython_code

if __name__ == "__main__":
    start_t = time.time()

    arr_a = np.arange(1000)
    arr_a = np.asarray(arr_a, dtype=np.complex)
    arr_b = np.arange(1000)
    arr_b = np.asarray(arr_b, dtype=np.complex)
    res = cython_code.cy_algo(arr_a, arr_b)
    print(res)

    all_time = time.time() - start_t
    print("Execution time:{0} [sec]".format(all_time))
