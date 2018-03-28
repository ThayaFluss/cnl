# CNL

### Cauchy Noise Loss
This is a tool for optimization of two random matrix models; compound Wishart model and information-plus noise model.

## Description
We consider an information plus noise random matrices as follows:
for a given N x N matrix A and a real number v > 0, define a random matrix
####    Y = A + vZ,

where Z is a (real or complex) Ginibre random matrix (i.e. whose entries are i.i.d. and distributed with N(0, sqrt(1/N)) ).
If the size N is large enough, eigenvalue distribution of Y^* Y can be approximated by  deterministic probability distribution on positive real line.

Our argorithm is based on the paper "On Cauchy noise loss in stohastic optimization of random matix models via free deterministic equivalents", to apper in arXiv.

## DEMO
In preperation

## Requirement
python 2 or 3, numpy, scipy, matplotlib. We recommend [Anaconda](https://www.continuum.io/downloads).

## Installation

```bash
$ git clone https://github.com/ThayaFluss/cnl.git
```

## Usage

To validate algorithms;
```bash
$ python validate_train_sc.py
$ python validate_train_cw.py
```

For the rank estimation of p x d matrix X;
```python
 X #numpy.array of shape [p,d]
 from rank_estimation import *
 rank, a, sigma = rank_estimation(X) #estimated rank and parameters  a, sigma.
```
For example;
'''python
 min_singular = 0.3
 true_rank = 30
 import numpy as np
 a_true = np.random.uniform(min_singular, 1, 50)
 from matrix_util import *
 A_true = rectangular_diag(a_true, p_dim=100, dim=50)
 X = info_plus_noise(A_true, sigma=0.1)
 rank, a, sigma = rank_estimation(X)
'''


## License

  [MIT](https://github.com/ThayaFluss/cnl/master/LICENSE)
