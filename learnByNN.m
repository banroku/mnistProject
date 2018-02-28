% === Learning By Logistic regression ===
J = 0;
lambda = 00;
iter = 001;

% === architecture of NN ==
K1 = 25;
K2 = 10;

% make new theta
if initializeTheta
    theta1 = randomInitTheta(zeros(K1, n + 1));
    theta2 = randomInitTheta(zeros(K2, K1 + 1));
    theta = [theta1(:); theta2(:)];
end

% %calculate initical J and theta
% [J_train, grad] = costLogisticReg(Xtrain, Ytrain, theta, lambda);

tic();
% train Logstic regressioa
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

