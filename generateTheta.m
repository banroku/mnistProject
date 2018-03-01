function theta = generateTheta(X, K)
    n = size(X, 1);
    depth = length(K);

    theta_i = randomInitTheta(zeros(K(1), n + 1));
    theta = theta_i(:);
    
    if depth > 1
        for i = 2:depth
            theta_i = randomInitTheta(zeros(K(2), K(1) + 1));
            theta = [theta; theta_i(:)];
        end
    end 
end

