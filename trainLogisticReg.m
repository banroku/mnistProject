function [theta] = trainLogisticReg(X, Y, theta_init, lambda, iter)
    
    costFunction = @(t) costLogisticReg(X, Y, t, lambda);
    options = optimset('MaxIter', iter, 'GradObj', 'on');
    theta = fmincg(costFunction, theta_init, options);
    
end
