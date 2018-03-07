function [J, Grad] = costGeneralizedNN(X, Y, k, Theta, lambda)
%function [J, Grad, Grad_math] = costGeneralizedNN(X, Y, k, Theta, lambda)
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

%    %unregulaized cost function
%    J = (1/m) * sum(sum( ...
%        (-Y .* log(h)) - (ones(size(Y)) - Y) .* log(ones(size(Y)) - h)...
%        ));

   %regulaized cost function
   JReg = 0;
   for i = 1:depth-1
       JReg = JReg + (lambda/(2 * m)) * sum(sum(theta{i}(:,2:end).^2));
   end
   J = (1/m) * sum(sum( ...
       (-Y .* log(h)) - (ones(size(Y)) - Y) .* log(ones(size(Y)) - h)...
       )) + JReg;

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

    %regularized gradiet
    for i = 1:depth-1
        grad{end-i} = (1/m) * D{end-i} + (lambda/m) * theta{end-i}; 
        grad{end-i}(:,1) = (1/m) * D{end-i}(:,1);
    end

%     %unregularized gradiet
%     for i = 1:depth-1
%         grad{end-i} = (1/m) * D{end-i}; 
%     end

%    regularized
%    grad2 = (1/m) * D2 + (lambda/m) * theta2; 
%    grad2(:,1) = (1/m) * D2(:,1);
%    grad1 = (1/m) * D1 + (lambda/m) * theta1; 

    for i = 1:depth-1
        Grad = [Grad; vec(grad{i})];
    end 

%     % == gradient checking == 
%     Grad_math = zeros(size(Grad));
%     delta = 0.0001;
%     thetaPlus = cell(depth,1);
% 
%     for j = 1:010
%         ThetaPlus = Theta;
%         ThetaPlus(j) = ThetaPlus(j) + delta;
% 
%         sep2 = 0;
%         for i = 1:depth-1
%              sep1 = sep2 + 1;
%              sep2 = sep2 + K(i+1) * ( K(i) + 1 );
%              thetaPlus{i} = reshape(ThetaPlus(sep1:sep2), K(i+1), K(i) + 1);
%         end
% 
%         %forward
%         zPlus = cell(depth, 1);
%         aPlus = cell(depth, 1);
%         aPlus{1} = [ones(1, m); X]; 
% 
%         for i = 1:depth-1
%             zPlus{i+1} = thetaPlus{i} * aPlus{i};
%             aPlus{i+1} = [ones(1, m); sigmoid(zPlus{i+1})];
%         end
%         hPlus = aPlus{depth}(2:end,:); %remove bias at final layer
% 
%         %unregulaized cost function
%         JPlus = (1/m) * sum(sum( ...
%             (-Y .* log(hPlus)) - (ones(size(Y)) - Y) .* log(ones(size(Y)) - hPlus)...
%             ));
% 
%         Grad_math(j) = (JPlus-J)/delta;
%     end

end
