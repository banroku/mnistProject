function [theta] = trainNN(X, Y, K1, K2, theta_init, lambda, iter)
    
    costFunction = @(t) costNN(X, Y, K1, K2, t, lambda);
    options = optimset('MaxIter', iter, 'GradObj', 'on');
    theta = fmincg(costFunction, theta_init, options);
    
end
