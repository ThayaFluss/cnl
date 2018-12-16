import numpy as np

class ISTA(object):
    """Iterative Shrinkage-Thresholding Algorithm"""
    def __init__(self, alpha):
        super(ISTA, self).__init__()
        if alpha < 0:
            raise  ValueError("alpha must be positive")

        self.alpha = alpha


    """
    if  x[i] < -alpha: return  x[i] + alpha
    if  x[i] > alpha : return x[i] - alpha
    else: return 0
    """
    def run(self, x):
        y =  abs(x)  - self.alpha
        y[y<0] = 0  ## y = y_+
        return y*np.sign(x)
