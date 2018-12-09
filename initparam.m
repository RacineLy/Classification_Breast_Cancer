% Function to initialize parameters of neural network
%-------------------------------------------------------------------------------
% Input     Lin     : Number of incoming data (numbers of colmuns)
% Output    Lout    : Number of outgoing data (number of lines)
%           epsilon : Parameter of randomization

function W = initparam(Lin, Lout, epsilon)
  
  W = rand(Lout, Lin+1)*(2*epsilon) - epsilon;
  
endfunction
