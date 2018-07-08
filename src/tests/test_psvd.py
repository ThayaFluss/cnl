import unittest
import numpy as np
from psvd import *
from random_matrices import *

class TestPSVD(unittest.TestCase):
    def test_psvd_cnl(self):
        p = 50
        d = 50
        sigma = 0.2
        zero_dim = 40
        min_singular = 0.3
        sample_mat = random_from_diag(p,d,zero_dim, min_singular)
        sample_mat += sigma*Ginibre(p,d)
        sample_mat *= 2
        U,D, V = psvd_cnl(sample_mat)
        #import pdb; pdb.set_trace()
