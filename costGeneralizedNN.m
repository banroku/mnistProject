function [J, Grad] = costGeneralizedNN(X, Y, k, Theta, lambda)
%function [J, grad, grad_math] = costNN1(X, Y, K1, K2, theta, lambda)
    m = size(X, 2);
    n = size(X, 1);
    K = [n; k];
    depth = size(K,1);
    Y = matrixizeY(Y, K(end));

    J = 0;
    theta = cell(depth,1);
    grad = cell(depth,1);
    Grad = [];

    %reshape theta (theta -> theta1, theta2)
    sep2 = 0;
    for i = 1:depth-1
        sep1 = sep2 + 1;
	sep2 = sep2 + K(i+1) * ( K(i) + 1 );
        theta{i} = reshape(Theta(sep1:sep2), K(i+1), K(i) + 1);
    end

    %forward
    z = cell(depth, 1);
    a = cell(depth, 1);
    a{1} = [ones(1, m); X]; 

    for i = 1:depth-1
        z{i+1} = theta{i} * a{i};
        a{i+1} = [ones(1, m); sigmoid(z{i+1})];
    end
    h = a{depth}(2:end,:); %remove bias at final layer

    %unregulaized
    J = (1/m) * sum(sum( ...
        (-Y .* log(h)) - (ones(size(Y)) - Y) .* log(ones(size(Y)) - h)...
        ));

%     regularized
%     J = (1/m) * sum(sum( ...
%         (-Y .* log(a3)) - (ones(size(Y)) - Y) .* log(ones(size(Y)) - a3) ...
%         )) + ...
%         (lambda/(2 * m)) * ( ...
%         sum(sum(theta1(:,2:end).^2)) + sum(sum(theta2(:,2:end).^2)) ...
%         );
% 

    %Backword
    d = cell(depth, 1);
    D = cell(depth, 1);
    d{end} = h - Y;
    for i = 1:depth-2
        d{end-i} = theta{end-i}(:,2:end)' * d{end+1-i} .* ...
                   sigmoidGradient(z{end-i});
    end

    for i = 1:depth-1
        D{end-i} = d{end+1-i} * a{end-i}';
    end

    for i = 1:depth-1
        grad{end-i} = (1/m) * D{end-i}; 
    end
%    regularized
%    grad2 = (1/m) * D2 + (lambda/m) * theta2; 
%    grad2(:,1) = (1/m) * D2(:,1);
%    grad1 = (1/m) * D1 + (lambda/m) * theta1; 

    for i = 1:depth-1
        Grad = [Grad; vec(grad{i})];
    end 

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
