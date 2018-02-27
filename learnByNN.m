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
theta = trainNN1(Xtrain, Ytrain, K1, K2, theta, lambda, iter);
trainingTime = toc();

% continue using trained theta
initializeTheta = false;

% calculate correct rate
Pre_train = predictNN1(Xtrain, K1, K2, theta);
Acc_train = calculateAccuracy(Ytrain, Pre_train);

% calculate parameters of cross-valication set
J_train = costNN1(Xtrain, Ytrain, K1, K2, theta, 0);
J_cv = costNN2(Xcv, Ycv, K1, K2, theta, 0);
Pre_cv= predictNN1(Xcv, K1, K2, theta);
Acc_cv = calculateAccuracy(Ycv, Pre_cv);

fprintf('Training time is: %f \n', trainingTime);
fprintf('Training time per iter: %f \n', trainingTime/iter);
fprintf('Cost of train set is: %f \n', J_train);
fprintf('Cost of cv set is: %f \n', J_cv);
fprintf('Accuracy of train set is: %f \n', Acc_train);
fprintf('Accuracy of cv set is: %f \n', Acc_cv);

% calculate costs





% === End: Logistic regression ===

