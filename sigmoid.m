% Function to compute sigmoid fonction for one neuron
%-------------------------------------------------------------------------------
% Input     Argument
% Output    Value of sigmoid function

function g = sigmoid(z)
  
  g = 1./(1 + exp(-z));
  
endfunction
