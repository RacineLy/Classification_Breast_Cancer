% Function to make prediction given neural network parameters
%-------------------------------------------------------------------------------
% Input : theta1 - Neural Network parameters : from input layer to hidden layer
%         theta2 - Neural Network parameters : from hidden layer to output layer
%         X      - Input Data

function p = predict(theta1, theta2, X)
  
  m = size(X,1);
  
  h1 = sigmoid([ones(m,1) X]*theta1');
  h2 = sigmoid([ones(m,1) h1]*theta2');
  [dummy p] = max(h2, [], 2);
  
  
endfunction
