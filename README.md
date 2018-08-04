# CNL

### Cauchy Noise Loss
This is a tool for optimization of two random matrix models; compound Wishart model and signal-plus-noise (information-plus-noise) model.

## Description
The compound Wishart model is
$W = Z^TAZ$,
where $Z$ is a (real or complex) $p \times  d$ Ginibre random matrix (i.e. whose entries are i.i.d. and distributed with $N(0, \sqrt(1/d)))$, and A is $p \times p$ deterministic self-adjoint matrix.


We consider a signal-plus-noise random matrices as follows:
for a given $p \times  d$  matrix $A$ and a real number $\sigma > 0$, define a random matrix
$$  Y = A + \sigma Z,\  W = Y^*Y $$

If $d$ is large enough and  $p= O(d)$, eigenvalue distribution of $Y^* Y$ can be approximated by  deterministic probability distribution on positive real line.

Our algorithm is based on the paper "Cauchy noise loss for stochastic optimization of random matrix models via free deterministic equivalents (https://arxiv.org/abs/1804.03154)".



## Requirement
python 3.6, numpy, scipy, matplotlib, tqdm.  We recommend to use the platform [Anaconda](https://www.continuum.io/downloads).

## Installation

```bash
$ git clone https://github.com/ThayaFluss/cnl.git
```
## Setup

```bash
$ bash setup.bash
cd src
python -m unittest tests/*.py
```

## Usage
```bash
cd src
```

For the probabilistic singular value decomposition based on Cauchy noise loss;
```python
 X #numpy.array of shape [p,d]
 from psvd import *
 U,D,V = psvd_cnl(X)
```


For the rank estimation of $p \times d$ matrix $X$;
```python
 X #numpy.array of shape [p,d]
 from psvd import *
 rank, a, sigma = rank_estimation(X) #estimated rank and parameters  a, sigma.
```



For example; (https://github.com/ThayaFluss/cnl/blob/master/demo_rank_estimation.py)
```python
 p_dim = 100
 dim = 50
 min_singular = 0.3
 true_rank = 30
 import numpy as np
 a_true = np.random.uniform(min_singular, 1, dim)
 for i in range(dim-true_rank):
   a_true[i] = 0  
 from matrix_util import *
 D = rectangular_diag(a_true, p_dim=p_dim, dim=dim)
 from random_matrices import *
 U = haar_unitary(p_dim)
 V = haar_unitary(dim)
 A_true = U @ D @ V ; #random rotation
 X = signal_plus_noise(A_true, sigma=0.1) ### sample matrix
 from psvd import *
 rank, a, sigma = rank_estimation(X) ### estimated rank and parameters
 print(rank, true_rank) ### compare with the true_rank !
 ```




## Validation

To validate algorithms in the same way as the numerical experiments in the paper;
```bash
cd src
$ python validate_train_sc.py
$ python validate_train_cw.py
```


## License

  [MIT](https://github.com/ThayaFluss/cnl/blob/master/LICENSE)
