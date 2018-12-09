% Main code to train a neural network to predict breast cancer.
% Neural network properties : 1 hidden layer 
% Source of Data            : ...
% Features
%    #  Attribute                     Domain
%   -- -----------------------------------------
%   1. Sample code number            id number
%   2. Clump Thickness               1 - 10
%   3. Uniformity of Cell Size       1 - 10
%   4. Uniformity of Cell Shape      1 - 10
%   5. Marginal Adhesion             1 - 10
%   6. Single Epithelial Cell Size   1 - 10
%   7. Bare Nuclei                   1 - 10
%   8. Bland Chromatin               1 - 10
%   9. Normal Nucleoli               1 - 10
%  10. Mitoses                       1 - 10
%  11. Class:                        (2 for benign, 4 for malignant)

%-------------------------------------------------------------------------------

% Reset environment
clear; clear all; clc;

% Read Data
Data = load('Data.txt');

% load shuffled Data
load Mat;
Mat = Mat(:,(2:end));

% Data size
[mData nData] = size(Mat);

% Replace classes with 1 (benign) or 2 (malignant)
ind_benign              = find(Mat(:,end) == 2);
ind_malignant           = find(Mat(:,end) == 4);
Mat(ind_benign,end)    = 1;
Mat(ind_malignant,end) = 2;

% Define training / validation and test dataset
indtrain = round(0.6*mData);
indvalid = round(0.2*mData);

% Training Dataset
Xtrain   = Mat(1:indtrain,1:(nData-1));
ytrain   = Mat(1:indtrain,end);
[m n]    = size(Xtrain);

% Validation Dataset
Xvalid   = Mat((indtrain+1):(indtrain+indvalid),1:(nData-1));
yvalid   = Mat((indtrain+1):(indtrain+indvalid),end);

% Test Dataset
Xtest    = Mat((indtrain+indvalid+1):end,1:(nData-1));
ytest    = Mat((indtrain+indvalid+1):end,end);

% Neural Network Parameters
n_inlayers  = size(Xtrain,2);
n_hidden    = 5;
n_outlayers = numel(unique(ytrain));

% Initialize Neural network parameters
epsilon   = 0.12;
theta1    = initparam(n_inlayers, n_hidden, epsilon);
theta2    = initparam(n_hidden, n_outlayers, epsilon);
nn_params = [theta1(:) ; theta2(:)];

% Compute cost
lambda = 1;
[cost grad] = nnCostFunction(nn_params, n_inlayers, n_hidden, n_outlayers, ...
                             Xtrain, ytrain, lambda);

% Compare gradient values with numerical approximation
checkNNGradients;

% Compute learning curves
%[error_train, error_valid] = learning_curves(nn_params, n_inlayers, n_hidden, ...
%                                             n_outlayers, Xtrain, ytrain, Xvalid, ...
%                                             yvalid, lambda);

% Compute Validation Curves
lambdavec                        = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3];
[error_train_vc, error_valid_vc] = validation_curves(nn_params, n_inlayers, n_hidden, ...
                                             n_outlayers, Xtrain, ytrain, Xvalid, ...
                                             yvalid, lambdavec);                                           

% Training Neural Network
options            = optimset('MaxIter',300);
u                  = find(error_valid_vc == min(error_valid_vc));
lambda             = lambdavec(u);
costfunc           = @(p) nnCostFunction(p, n_inlayers, n_hidden, n_outlayers, ...
                                         Xtrain, ytrain, lambda);
[nn_weights, cost] = fmincg(costfunc, nn_params, options);

% Roll parameters
theta1_nn = reshape(nn_weights(1:(n_hidden*(n_inlayers+1))), n_hidden, (n_inlayers+1));
theta2_nn = reshape(nn_weights(((n_hidden*(n_inlayers+1))+1):end), n_outlayers, (n_hidden+1));

% Fitting Accuracy
pred_fit = predict(theta1_nn, theta2_nn, Xtrain);
accu_fit = mean(double(pred_fit == ytrain))*100;

% Accuracy on test set
pred     = predict(theta1_nn, theta2_nn, Xtest);
accuracy = mean(double(pred == ytest))*100;

% Plot Learning Curves
plotData;  

