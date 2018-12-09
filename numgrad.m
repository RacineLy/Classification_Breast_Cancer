% Function to compute gradient using a numerical approximation
%-------------------------------------------------------------------------------
% Input  :     J  : Cost function
%           theta : Unrolled parameters
% Output :  numerical gradient using central finite difference methods

function g = numgrad(J, theta)
  
  perturb = zeros(size(theta));
  g       = zeros(size(theta));
  e       = 1e-4;
  
  for i = 1:numel(theta)
    
    perturb(i) = e;
    loss1      = J(theta - perturb);
    loss2      = J(theta + perturb);
    g(i)       = (loss2 - loss1)/(2*e);
    perturb(i) = 0;
    
  end
  
   
endfunction
