from utils.schedulers import *

import unittest


class TestSchedulers(unittest.TestCase):
    """docstring for TestSchedulers."""
    def test_fix(self):
        t = 100
        s = Fix()
        assert s.get(t) == 1

    def test_inv(self):
        t = 100
        s = Inv()
        mult = s.get(t)


    def test_step(self):
        t = 100
        stepsize = 10
        decay = 0.1
        s = Step(stepsize, 0.1)
        mult = s.get(t)
        assert mult == decay ** (-0.1)
