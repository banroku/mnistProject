% === Reset and Set-up environment to start mnist project
clear all; close all; clc
load "mymnist.m";
K = 10;
initializeTheta = true;

%Transpose Xs
Xtrain = Xtrain'; Xcv = Xcv'; Xtest = Xtest';

%convert 0 to 10 in Y
Ytrain = Ytrain + (Ytrain==0)*10;
Ycv = Ycv + (Ycv==0)*10;
Ytest = Ytest + (Ytest==0)*10;

%make 01 matrix of y
%Ytrain = matrixizeY(Ytrain, K);
%Ycv = matrixizeY(Ycv, K);
%Ytest = matrixizeY(Ytest, K);

%check row-column true 
if (checkRC(Xtrain, Xcv) == false)
    fprintf('Row and column of X maybe conversed.');
endif
if (checkRC(Ytrain, Ycv) == false)
    fprintf('Row and column of Y maybe conversed.');
endif

%feature scaling
[Xtrain Xcv Xtest mu range] = featureScaling(Xtrain, Xcv, Xtest);

%shuffling train data for stocastic decent
[Xtrain Ytrain] = orderShuffling(Xtrain, Ytrain);


%other useful constands
m = size(Xtrain, 2);
n = size(Xtrain, 1);
%end
