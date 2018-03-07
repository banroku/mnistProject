function [p, p_rate] = predictGeneralizedNN(X, k, Theta)
    m = size(X,2);
    n = size(X,1);
    K = [n; k];
    depth = size(K,1);
    theta = cell(depth,1);

    %reshape theta (theta -> theta1, theta2)
    sep2 = 0;
    for i = 1:depth-1
        sep1 = sep2 + 1;
	sep2 = sep2 + K(i+1) * ( K(i) + 1 );
        theta{i} = reshape(Theta(sep1:sep2), K(i+1), K(i) + 1);
    end

    p = zeros(1, m);
    p_rate = zeros(1, m);
 
    %forward
    z = cell(depth, 1);
    a = cell(depth, 1);
    a{1} = [ones(1, m); X]; 

    for i = 1:depth-1
        z{i+1} = theta{i} * a{i};
        a{i+1} = [ones(1, m); sigmoid(z{i+1})];
    end
    h = a{depth}(2:end,:); %remove bias at final layer

    [p_rate, p] = max(h);

end
