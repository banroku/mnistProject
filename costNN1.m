function [J, grad] = costNN1(X, Y, K1, K2, theta, lambda)
%cost function [J, grad] = costNN1(X, Y, K1, K2, theta, lambda)
    m = size(X, 2);
    n = size(X, 1);
    Y = matrixizeY(Y, K2);

    %reshape theta (theta -> theta1, theta2)
    theta1 = reshape(theta(1: K1*(n+1)), K1, n+1);
    theta2 = reshape(theta((K1*(n+1) + 1): size(theta)), K2, K1 + 1);

    X = [ones(1, m); X];
    J = 0;
    grad1 = zeros(size(theta1));
    grad2 = zeros(size(theta2));

    z1 = theta1 * X;
    a1 = sigmoid(z); %use sigmoid?
    z2 = theta2 * [ones(1, K1); a1];
    a2 = sigmoid(z2);

%    % w/o regulation
%    J = 
%    grad1 = 
%    grad2 = 
%
%    grad = [grad1(:) grad2(:)];
%
%    % == gradient checking ==



end
