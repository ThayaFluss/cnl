function [out] = test_base_iter(num_iter, st)
%TEST_BASE_ITER Summary of this function goes here
%   Detailed explanation goes here

A = rand(2,2)
A = reshape(A, [2,2])

for i = 1:num_iter
    C= A*A
end

out = A

out = 1

end
