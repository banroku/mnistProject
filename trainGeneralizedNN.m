function [theta] = trainGeneralizedNN(X, Y, K, theta_init, lambda, iter)
    
    costFunction = @(t) costGeneralizedNN(X, Y, K, t, lambda);
    options = optimset('MaxIter', iter, 'GradObj', 'on');
    theta = fmincg(costFunction, theta_init, options);
    
end
