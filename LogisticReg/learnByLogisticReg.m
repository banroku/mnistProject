% === Learning By Logistic regression ===
J = 0;
lambda = 10;
iter = 010;

% make new theta
if initializeTheta
    theta = zeros(K, size(Xtrain, 1) + 1);
    theta = theta(:);
end

% %calculate initical J and theta
% [J_train, grad] = costLogisticReg(Xtrain, Ytrain, theta, lambda);

tic();
% train Logstic regression
theta = trainLogisticReg(Xtrain, Ytrain, theta, lambda, iter);
trainingTime = toc();

% continue using trained theta
initializeTheta = false;

% calculate correct rate
Pre_train = predictLogisticReg(Xtrain, theta);
Acc_train = calculateAccuracy(Ytrain, Pre_train);

% calculate parameters of cross-valication set
J_train = costLogisticReg(Xtrain, Ytrain, theta, 0);
J_cv = costLogisticReg(Xcv, Ycv, theta, 0);
Pre_cv= predictLogisticReg(Xcv, theta);
Acc_cv = calculateAccuracy(Ycv, Pre_cv);

fprintf('Train time (per iter): %f (%f) \n', trainingTime, trainingTime/iter);
fprintf('Cost (train, cv): %f, %f \n', J_train, J_cv);
fprintf('Accuracy (train, cv): %f, %f \n', Acc_train, Acc_cv);


% === End: Logistic regression ===

