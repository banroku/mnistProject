% === Learning By Logistic regression ===
J = 0;
lambda = 000;
iter = 005;

% === architecture of NN ==
K_NN = [25; 10];
K1 = 25;
K2 = 10;

% make new theta
if initializeTheta
    theta = generateTheta(Xtrain, K_NN);
end

tic();

% % train by NN
% theta = trainNN(Xtrain, Ytrain, K1, K2, theta, lambda, iter);
% train by generalized NN
theta = trainNN(Xtrain, Ytrain, K1, K2, theta, lambda, iter);

trainingTime = toc();

% continue using trained theta
initializeTheta = false;

% calculate correct rate
Pre_train = predictNN(Xtrain, K1, K2, theta);
Acc_train = calculateAccuracy(Ytrain, Pre_train);

% calculate parameters of cross-valication set
J_train = costNN(Xtrain, Ytrain, K1, K2, theta, 0);
J_cv = costNN(Xcv, Ycv, K1, K2, theta, 0);
Pre_cv= predictNN(Xcv, K1, K2, theta);
Acc_cv = calculateAccuracy(Ycv, Pre_cv);

fprintf('Train time (per iter): %f (%f) \n', trainingTime, trainingTime/iter);
fprintf('Cost (train, cv): %f, %f \n', J_train, J_cv);
fprintf('Accuracy (train, cv): %f, %f \n', Acc_train, Acc_cv);

% calculate costs





% === End: Logistic regression ===

