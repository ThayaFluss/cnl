import unittest
import numpy as np
from psvd import *
from random_matrices import *

class TestPSVD(unittest.TestCase):
    def test_prob_svd_cnl(self):
        p = 50
        d = 50
        sigma = 0.5
        zero_dim = 0
        min_singular = 0
        sample_mat = random_from_diag(p,d,zero_dim, min_singular)
        sample_mat += sigma*Ginibre(p,d)
        U,D, V = prob_svd_cnl(sample_mat)
