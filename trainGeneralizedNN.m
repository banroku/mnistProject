function [theta] = trainGeneralizedNN(X, Y, K, theta_init, lambda, iter, batchSize)
    m = size(X, 2);
    batchNo = floor(m/batchSize);
    
    for j = 1:iter
        for i = 1:batchNo
            Xbatch = X(:, batchSize*(i-1)+1:batchSize*i);
            Ybatch = Y(:, batchSize*(i-1)+1:batchSize*i);
            costFunction = @(t) costGeneralizedNN(Xbatch, Ybatch, K, t, lambda);
            options = optimset('MaxIter', 1, 'GradObj', 'on');
            theta = fmincg(costFunction, theta_init, options);
        end
    end
end
