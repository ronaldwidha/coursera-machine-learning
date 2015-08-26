function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% PART 1
% ======

% cost
% ----

function a = hypothesis(xi, theta)
    z = xi * theta';
    a = sigmoid(z);
end

% calculate hypothesis - this is flexible can handle more layers
thetas = { Theta1, Theta2 };
gzs = {};

% add bias
gz = [ones(size(X,1),1) X];
gzs{1} = gz;

for l=1:(size(thetas, 2))
    
    gz = hypothesis(gz, thetas{l}); 
        
    % add bias if not output layer
    if l != size(thetas,2) 
        gz = [ones(size(gz,1),1) gz];
    endif
    
    % store for later
    gzs{l+1} = gz;
end
hx = gz;

% convert label into output network
ymatrix = zeros(m, num_labels); 
for k = 1:num_labels
    ymatrix(:, k) = (y == k);
endfor

% calculate cost
for i =1:m
    J += sum( -ymatrix(i,:) * log(hx(i,:)') - ((1.-ymatrix(i,:)) * log(1-hx(i,:)')));
endfor

J /= m;

% cost with regularization
% ------------------------

% take off the bias
thetasWithoutBias = {};
for l=1:(size(thetas, 2))
    thetasWithoutBias{l} = thetas{l}(:,2:end);    
endfor

Regularization = 0;
for l=1:(size(thetasWithoutBias, 2))
    % square. then sum up the row. then sum up the columns
    Regularization = Regularization + sum( sum(thetasWithoutBias{l}.^2,1), 2);
endfor

J += lambda / (2 * m) * Regularization;



% PART 2
% ======

delta = {};
Delta = {zeros(size(Theta1)), zeros(size(Theta2))};

for t=1:m 
    % Point 1 - put bias on every single data
    a1 = [1 ; X(t, :)'];
    z2 = Theta1 * a1;
        
    a2 = [1 ; sigmoid(z2)];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    
    % point 2
    delta{3} = zeros(num_labels,1);
    
    for k=1:num_labels
        delta{3}(k) = a3(k) - ymatrix(t,k);
    endfor
    
    % point 3
    delta{2} =  Theta2' * delta{3} .* sigmoidGradient([1; z2]); 
    
    % point 4
    % get rid of delta for bias
    delta{2} = delta{2}(2:end);
    
    Delta{1} += delta{2} * (a1'); 
    Delta{2} += delta{3} * (a2');
 
endfor

% gradient
% --------

Theta1_grad = 1 / m .* Delta{1};
Theta2_grad = 1 / m .* Delta{2};

% regularized gradient
% ---------------------

% update only 2nd column and beyond since, 1st column is weighting for bias
% also remove delta from bias
Theta1_grad(:,2:end)= 1 / m .* Delta{1}(:,2:end) + lambda / m .* Theta1(:,2:end); 
Theta2_grad(:,2:end)= 1 / m .* Delta{2}(:,2:end) + lambda / m .* Theta2(:,2:end); 

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
