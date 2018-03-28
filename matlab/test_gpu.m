function test_gpu()
    dim = 100;
    p_dim = dim;
    size = 2*dim
    sample = 32
    i_iter = 20
    b_iter = 5
    A = rand(size,size) + sqrt(-1)*eye(size);
    A_gpu  =  gpuArray(A);
    %tic
    %for k = 1:sample
    %    for i =1:i_iter
    %      G = inv(A_gpu);
    %    end
    %end
    %toc
    AA_gpu = gpuArray(rand(size,size,sample));
    tic
    for i = 1:i_iter
     G = pagefun(@inv, AA_gpu);
    end
    toc
    tic
    for k = 1:p_dim %param
         for i = 1:b_iter %iter for gradients
             grad = pagefun(@mtimes, AA_gpu,AA_gpu);
             abs_ = pagefun(@abs, grad);
             sum_ = bsxfun(@sum, abs_);
         end
    end
    toc
end
