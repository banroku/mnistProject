function [theta] = trainLogisticReg(X, Y, theta_init, lambda)
    
    costFunction = @(t) costLogisticReg(X, Y, t, lambda);
    options = optimset('MaxIter', 001, 'GradObj', 'on');
    theta = fmincg(costFunction, theta_init, options);
    
end
