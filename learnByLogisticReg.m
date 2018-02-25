% === Learning By Logistic regression ===
J = 0;
lambda = 1;
iter = 50;

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
Pre_train = predict(Xtrain, theta);
Acc_train = calculateAccuracy(Ytrain, Pre_train);

% calculate parameters of cross-valication set
J_train = costLogisticReg(Xtrain, Ytrain, theta, 0);
J_cv = costLogisticReg(Xcv, Ycv, theta, 0);
Pre_cv= predict(Xcv, theta);
Acc_cv = calculateAccuracy(Ycv, Pre_cv);

fprintf('Training time is: %f \n', trainingTime);
fprintf('Training time per iter: %f \n', trainingTime/iter);
fprintf('Cost of train set is: %f \n', J_train);
fprintf('Cost of cv set is: %f \n', J_cv);
fprintf('Accuracy of train set is: %f \n', Acc_train);
fprintf('Accuracy of cv set is: %f \n', Acc_cv);

% calculate costs





% === End: Logistic regression ===

