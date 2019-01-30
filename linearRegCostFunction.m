function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

h_theta = X * theta;

%Cost function without reg
J = (1/(2*m)) * ((h_theta - y)' * (h_theta - y));

%Reg
J = J + (lambda/(2*m)) * (theta(2:end)' * theta(2:end));

%Grad
grad = (1/m) * X' * (h_theta - y);
grad(2:end) = grad(2:end) + lambda/m * theta(2:end);

grad = grad(:);

end
