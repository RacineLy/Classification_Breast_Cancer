% Function to compute gradient of sigmoid
%-------------------------------------------------------------------------------
% Input     Argument
% Output    Value of the graditn of the sigmoid

function g = gradsigmoid(z)
  
  var = sigmoid(z);
  g   = var.*(1 - var);
  
endfunction
