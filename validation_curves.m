% Function to compute validation curves
% ------------------------------------------------------------------------------
% Input : nnparams    > Unroll neural network parameters
%         n_inlayers  > Number of neurons at input layer
%         n_hidden    > Number of neurons at hidden layer
%         n_outlayers > Number of neurons at output layer
%         Xtrain      > Training DataSet Features
%         ytrain      > Training DataSet labels
%         Xvalid      > Validation DataSet Features
%         yvalid      > Validation DataSet labels
%         lambdavec   > vector of regularization values.




function [error_train, error_valid] = validation_curves(nn_params, n_inlayers, ...
                                                        n_hidden, n_outlayers, ...
                                                        X, y, Xvalid, yvalid, lambdavec);
                                                      
m           = size(X,1);
error_train = zeros(numel(lambdavec),1);
error_valid = zeros(numel(lambdavec),1);
options     = optimset('MaxIter', 100);

for t = 1:numel(lambdavec)
  
  lambda          = lambdavec(t);
  
  costfunc        = @(p) nnCostFunction(p, n_inlayers, n_hidden, n_outlayers, ...
                                        X, y, lambda);
                                 
  [weights, cost] = fmincg(costfunc, nn_params, options);

  error_train(t)  = nnCostFunction(weights, n_inlayers, n_hidden, n_outlayers,...
                                   X, y, 0);
                                 
  error_valid(t)  = nnCostFunction(weights, n_inlayers, n_hidden, n_outlayers, ...
                                   Xvalid, yvalid, 0);  
                                   
  disp([(t/m)*100 error_train(t) error_valid(t)]);

end  
                  
                  
endfunction