function [out] = test_parallel()
%TEST_PARALLEL Summary of this function goes here
%   Detailed explanation goes here
size = 2
A = rand(size,size);
a = squeeze(reshape(A, [],size*size))
ndims(a)
A_gpu = gpuArray(a);

%4-sample
iters = [10,20,30,40];
As = [A_gpu,A_gpu,A_gpu,A_gpu];
sts = ['a', 'b', 'c', 'd']
C = arrayfun(@test_base_iter,iters, sts);

out = C

end

