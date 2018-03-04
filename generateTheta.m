function Theta = generateTheta(X, k)
    n = size(X, 1);
    K = [n; k];
    depth = length(K);
    Theta = [];
    theta = cell(K,1);

    for i = 1:depth-1
        theta{i} = randomInitTheta(zeros(K(i+1), K(i) + 1));
        Theta = [Theta ; theta{i}(:) ];
    end

end

