% Function to compute cost and gradient of neural network
%-------------------------------------------------------------------------------
% Input :   
%           nnparams    : Unrolled neural network parameters
%           n_inlayers  : number of input neurons
%           n_hidden    : number of hidden neurons
%           n_outlayers : number of output neurons
%           X           : Training DataSet
%           y           : Corresponding label for X
%           lambda      : Regularization parameter
%
% Output : 
%           cost        : cost function evaluation
%           gradient    : gradient values (using backpropagation)

function [cost, grad] = nnCostFunction(nn_params, n_inlayers, n_hidden, ...
                                       n_outlayers, X, y, lambda)

% Size of training dataset and design matrix                                       
m = size(X,1);
X = [ones(m,1) X];

% Unroll parameters
theta1 = reshape(nn_params(1:(n_hidden*(n_inlayers+1))), n_hidden, (n_inlayers+1));
theta2 = reshape(nn_params(((n_hidden*(n_inlayers+1))+1):end), n_outlayers, (n_hidden+1));

% Initialize neural network parameters
theta1nn = zeros(size(theta1));
theta2nn = zeros(size(theta2));

% Feed Forward propagation to compute cost
a1      = X';
z2      = theta1*a1;
a2      = sigmoid(z2);
a2      = [ones(1,m); a2];
z3      = theta2*a2;
h_theta = sigmoid(z3);

% Transform classes with binary values
ynew = zeros(n_outlayers,m);

for t = 1:m
  ynew(y(t),t) = 1;
end

% Compute regularization terms
t1  = theta1(2:end).^2;
t2  = theta2(2:end).^2;
Reg = (lambda/(2*m))*(sum(sum(t1)) + sum(sum(t2)));

% Compute hypothesis
hyp  = (1/m)*sum(sum(-ynew.*log(h_theta) - (1-ynew).*log(1 - h_theta)));

% Compute cost
cost = hyp + Reg;

% Compute gradient
% Feed Forward

for i = 1:m
  
  % Step 1 - Feed Forward
  a1 = X(i,:)';
  z2 = theta1*a1;
  a2 = sigmoid(z2);
  a2 = [1; a2];
  z3 = theta2*a2;
  a3 = sigmoid(z3);
  
  % Step 2 - Compute Error at output layer
  delta3 = a3 - ynew(:,i);
  
  % Step 3 - BackPropagation
  delta2 = (theta2)'*delta3.*gradsigmoid([1; z2]);
  delta2 = delta2(2:end);
  
  % Step 4 - Compute gradient accumulator
  theta1nn = theta1nn + (delta2)*(a1)';
  theta2nn = theta2nn + (delta3)*(a2)';
  
end

% Non regularization for first column terms
theta1nn = (1/m)*theta1nn;
theta2nn = (1/m)*theta2nn;

% Regularization from column 2 to end
theta1nn(:,(2:end)) = theta1nn(:,(2:end)) + (lambda/m)*theta1nn(:,(2:end));
theta2nn(:,(2:end)) = theta2nn(:,(2:end)) + (lambda/m)*theta2nn(:,(2:end));

grad = [theta1nn(:) ; theta2nn(:)];
            
endfunction
