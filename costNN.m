function [J, grad] = costNN(X, Y, K1, K2, theta, lambda)
%function [J, grad, grad_math] = costNN1(X, Y, K1, K2, theta, lambda)
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

    z2 = theta1 * X;
    a2 = [ones(1, m); sigmoid(z2)]; 
    z3 = theta2 * a2;
    a3 = sigmoid(z3);

    % w/o regulation
    J = (1/m) * sum(sum( ...
        (-Y .* log(a3)) - (ones(size(Y)) - Y) .* log(ones(size(Y)) - a3)...
        ));

    d3 = a3 - Y;
    d2 = theta2(:,2:end)' * d3 .* sigmoidGradient(z2);
    D2 = d3 * a2';
    D1 = d2 * X';
    grad2 = (1/m) * D2; 
    grad1 = (1/m) * D1;

    grad = [grad1(:); grad2(:)];

%     % == gradient checking == 
%     grad_math = zeros(size(grad));
%     delta = 0.0001;
%     theta_vect = theta;
%     for i = 1:100
%         thetaPlus = theta;
%         thetaPlus(i) = thetaPlus(i) + delta;
% 
%         theta1 = reshape(thetaPlus(1: K1*(n+1)), K1, n+1);
%         theta2 = reshape(thetaPlus((K1*(n+1) + 1): size(thetaPlus)), K2, K1 + 1);
% 
%         z2Plus = theta1 * X;
%         a2Plus = [ones(1, m); sigmoid(z2Plus)]; 
%         z3Plus = theta2 * a2Plus;
%         a3Plus = sigmoid(z3Plus);
% 
%         JPlus = (1/m) * sum(sum( ...
%             (-Y .* log(a3Plus)) - ...
%             (ones(size(Y)) - Y) .* log(ones(size(Y)) - a3Plus)...
%             ));
% 
%         grad_math(i) = (JPlus-J)/delta;
%     end

end