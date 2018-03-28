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
```bash
$ python validate_train_sc.py
$ python validate_train_cw.py
```

## License

  [MIT](https://github.com/ThayaFluss/cnl/master/LICENSE)
