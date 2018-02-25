% === Learning By Logistic regression ===
J = 0;
lambda=0;

% make new theta
if initializeTheta
    theta = zeros(K, size(Xtrain, 1) + 1);
    theta = theta(:);
end

%calculate initical J and theta
[J_train, grad] = costLogisticReg(Xtrain, Ytrain, theta, lambda);

% train Logstic regression
theta = trainLogisticReg(Xtrain, Ytrain, theta, lambda);

% continue using trained theta
initializeTheta = false;

% calculate correct rate
Pre_train = predict(Xtrain, theta);
Acc_train = calculateAccuracy(Ytrain, Pre_train);

% calculate parameters of cross-valication set
J_cv = costLogisticReg(Xcv, Ycv, theta, 0);
Pre_cv= predict(Xcv, theta);
Acc_cv = calculateAccuracy(Ycv, Pre_cv);

fprintf('Accuracy of train set is: %f \n', Acc_train);
fprintf('Accuracy of cv set is: %f \n', Acc_cv);

% calculate costs





% === End: Logistic regression ===

