function [p, p_rate] = predict(X, theta)
    m = size(X,2);
    n = size(X,1);
    K = size(theta) / (n + 1);

    theta = reshape(theta, K, n + 1);
    p = zeros(1, m);
    p_rate = zeros(1, m);

    X = [ones(1, m); X];

    Y = sigmoid(theta * X);

    [p_rate, p] = max(Y);
    
end
