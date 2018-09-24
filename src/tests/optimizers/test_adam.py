import unittest
import optimizers
from utils.schedulers import *
from optimizers.adam import Adam
import numpy as np

class TestAdam(unittest.TestCase):
    def test_update(self):
        Optimizer = Adam()
        Scheduler = Fix()

        param = np.zeros(100)
        Optimizer.setup(param, Scheduler)
        grad = np.zeros(100)
        Optimizer.update(param, grad)
