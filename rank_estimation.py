from train_fde import *
from env_logger import *

def rank_estimation(sample_mat,\
 base_scale = 2e-1, dim_cauchy_vec=2, base_lr = 0.1,minibatch_size=1,\
 max_epoch=120, normalize_sample = False,\
 reg_coef = 2e-4,\
 list_zero_thres=[1e-3]):

 p_dim = sample_mat.shape[0]
 dim = sample_mat.shape[1]
 _, sample, _ = np.linalg.svd(sample_mat)

 diag_A, sigma, _, _, num_zero_array = train_fde_sc(
 p_dim = p_dim, dim=dim, sample = sample,\
  base_scale = base_scale, dim_cauchy_vec=dim_cauchy_vec, base_lr = base_lr,minibatch_size=minibatch_size,\
  max_epoch=max_epoch, normalize_sample = normalize_sample,\
  reg_coef = reg_coef,\
  list_zero_thres=list_zero_thres )

 logging.info("list_zero_thres= {}".format( list_zero_thres))
 estimaed_ranks = dim - num_zero_array[-1]

 return estimaed_ranks,diag_A, sigma, 
