% === Learning By Logistic regression ===
J = 0;
lambda = 050;
iter = 100;

% === architecture of NN ==
K = [25; 100; 10];
K1 = K(1);
K2 = K(2);

% make new theta
if initializeTheta
    theta = generateTheta(Xtrain, K);
end

tic();

% % train by NN
% theta = trainNN(Xtrain, Ytrain, K1, K2, theta, lambda, iter);
% train by generalized NN
theta = trainGeneralizedNN(Xtrain, Ytrain, K, theta, lambda, iter);

trainingTime = toc();

% continue using trained theta
initializeTheta = false;

% calculate correct rate
Pre_train = predictGeneralizedNN(Xtrain, K, theta);
Acc_train = calculateAccuracy(Ytrain, Pre_train);

% calculate parameters of cross-valication set
J_train = costGeneralizedNN(Xtrain, Ytrain, K, theta, 0);
J_cv = costGeneralizedNN(Xcv, Ycv, K, theta, 0);
Pre_cv= predictGeneralizedNN(Xcv, K, theta);
Acc_cv = calculateAccuracy(Ycv, Pre_cv);

fprintf('Train time (per iter): %f (%f) \n', trainingTime, trainingTime/iter);
fprintf('Cost (train, cv): %f, %f \n', J_train, J_cv);
fprintf('Accuracy (train, cv): %f, %f \n', Acc_train, Acc_cv);

% calculate costs





% === End: Logistic regression ===

