function [J, grad] = costLogisticReg(X, Y, theta, lambda)
%function [J, grad, grad_math] = costLogisticReg(X, Y, theta, lambda)
    m = size(X, 2);
    n = size(X, 1);
    K = size(theta)/(n+1);
    Y = matrixizeY(Y, size(theta)/(n+1));
    %reshape theta 
    theta = reshape(theta, size(theta)/(n + 1), n + 1);

    X = [ones(1, m); X];
    J = 0;
    grad = zeros(size(theta));
    
    z = theta * X;
    a = sigmoid(z);
    
    % % w/o regulation
    % J = (1/m) * sum(sum(Y .* log(a) + not(Y) .* log(ones(size(a))-a)));
    % grad = (1/m) * (a - Y) * X';
    % w/o regulation
    J = (1/m) * sum(sum(-Y .* log(a) - not(Y) .* log(ones(size(a))-a))) ...
        + (lambda/ (2 * m)) * sum(sum(theta(:,2:n))) ;
    
    grad = (1/m) * (a - Y) * X' + (lambda/m) * theta;
    grad(:,1) = (1/m) * (a - Y) * X(1,:)';

    %unroll grad
    grad = grad(:);


%     % == gradient checking == 
%     grad_math = zeros(size(grad));
%     delta = 0.0001;
%     theta_vect = theta(:);
%     for i = 1:100
%         thetaPlus_vect = theta_vect;
%         thetaPlus_vect(i) = thetaPlus_vect(i) + delta;
%         thetaPlus = reshape(thetaPlus_vect, K, n+1);
%         aPlus = sigmoid(thetaPlus * X);
%         J0= (1/m) * sum(sum(-Y .* log(a) ...
%                 - not(Y) .* log(ones(size(a))-a)));
%         JPlus = (1/m) * sum(sum(-Y .* log(aPlus) ...
%                 - not(Y) .* log(ones(size(aPlus))-aPlus)));
%         grad_math(i) = (JPlus-J0)/delta;
%     end

end

