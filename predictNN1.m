function [p, p_rate] = predictNN1(X, K1, K2, theta)
    m = size(X,2);
    n = size(X,1);

    %reshape theta (theta -> theta1, theta2)
    theta1 = reshape(theta(1: K1*(n+1)), K1, n+1);
    theta2 = reshape(theta((K1*(n+1) + 1): size(theta)), K2, K1 + 1);

    p = zeros(1, m);
    p_rate = zeros(1, m);
 
    X = [ones(1, m); X];

    z2 = theta1 * X;
    a2 = sigmoid(z2); 
    z3 = theta2 * [ones(1, K1); a2];
    a3 = sigmoid(z3);

    [p_rate, p] = max(a3);
    
end
