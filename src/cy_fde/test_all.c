#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <cblas.h>

#include "test_matrix_util.h"


int test_all(int result){
  result |= test_blas(result);
  result |= test_my_zgemm(result);
  return result;
}
