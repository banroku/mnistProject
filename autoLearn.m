% === Learning By Logistic regression ===

K_screen = cell(4,1);
K_screen{1} = [25; 10];
K_screen{2} = [100; 10];
K_screen{3} = [100; 25; 10];
K_screen{4} = [25; 100; 10];
lambda_screen = [0; 0.0001; 0.0003; 0.001; 0.003; 0.01; 0.03; 0.1; 0.3; 1];

result = zeros(4*length(lambda_screen), 7);

for i = 1:4
    initializeTheta = 1;
    K = K_screen{i};

    for j = 1:length(lambda_screen)
        iter = 010;
        batchSize = 00100;
        J = 0;
        lambda = lambda_screen(j);

        % make new theta
        if initializeTheta
            theta_init = generateTheta(Xtrain, K);
        end

        tic();
        theta = trainGeneralizedNN(Xtrain, Ytrain, K, theta_init, lambda, iter, batchSize);
        
        trainingTime = toc();
        
        % continue using trained theta
        initializeTheta = false;
        
        % calculate correct rate
        J_train = costGeneralizedNN(Xtrain, Ytrain, K, theta, 0);
        Pre_train = predictGeneralizedNN(Xtrain, K, theta);
        Acc_train = calculateAccuracy(Ytrain, Pre_train);
        J_cv = costGeneralizedNN(Xcv, Ycv, K, theta, 0);
        Pre_cv= predictGeneralizedNN(Xcv, K, theta);
        Acc_cv = calculateAccuracy(Ycv, Pre_cv);
	result((i-1) * length(lambda_screen) + j, :) = ...
            [i lambda J_train J_cv Acc_train Acc_cv trainingTime/iter];
        save result.mat result;
    end
end
