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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% linear regression - do not need sigmoid activation
hx =  X * theta;

% do not normalize the first feature
theta1toend = theta; 
theta1toend(1) = 0; 

delta = hx-y;

J = 1 / (2 * m) * ( delta' * delta ) + lambda / (2*m) * sum(theta1toend.^2);

% for j = 0 do not normalize, so use theta1toend
grad = grad + (1 / m * X' * delta) + ((lambda / m) .* theta1toend);

% =========================================================================

grad = grad(:);

end
