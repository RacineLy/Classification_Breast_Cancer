% Function to compute learning curves
% ------------------------------------------------------------------------------
% Input : nnparams    > Unroll neural network parameters
%         n_inlayers  > Number of neurons at input layer
%         n_hidden    > Number of neurons at hidden layer
%         n_outlayers > Number of neurons at output layer
%         Xtrain      > Training DataSet Features
%         ytrain      > Training DataSet labels
%         Xvalid      > Validation DataSet Features
%         yvalid      > Validation DataSet labels




function [error_train, error_valid] = learning_curves(nn_params, n_inlayers, ...
                                                      n_hidden, n_outlayers, ...
                                                      X, y, Xvalid, yvalid, lambda);
                                                      
m           = size(X,1);
error_train = zeros(m,1);
error_valid = zeros(m,1);
options     = optimset('MaxIter', 100);

for t = 1:m
  
  costfunc = @(p) nnCostFunction(p, n_inlayers, n_hidden, n_outlayers, ...
                                 X((1:t),:), y(1:t), lambda);
                                 
  [weights, cost] = fmincg(costfunc, nn_params, options);

  error_train(t)  = nnCostFunction(weights, n_inlayers, n_hidden, n_outlayers,...
                                   X((1:t),:), y(1:t), 0);
                                 
  error_valid(t)  = nnCostFunction(weights, n_inlayers, n_hidden, n_outlayers, ...
                                   Xvalid, yvalid, 0);  
                                   
  disp([(t/m)*100 error_train(t) error_valid(t)]);

end  
                  
                  
endfunction
