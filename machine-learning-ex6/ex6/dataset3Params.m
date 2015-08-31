function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%possibleValues = { 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30 };
%minerror = 10000000;

% display('this is going to take a while');

% brute force
% though by intuition, it needs to be around the middle values.

%for i=1:size(possibleValues,2)
%	for j=1:size(possibleValues,2)
		
%		C = possibleValues{i};
%		sigma = possibleValues{j};
%
%		%disp("C:"), disp(C);
%		%disp("sigma:"), disp(sigma);
%		
%		model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
%		predictions = svmPredict(model, Xval);
%		error = mean(double(predictions ~= yval));
%
%		disp(error);
%
%		if error < minerror
%			minerror = error;
%			minC = C;
%			minSigma = sigma;
%		endif
%	endfor
%endfor

% to speed up execution time
sigma = 0.1 % minSigma;
C = 1 % minC;

%display("optimal sigma"), display(sigma);
%display("optimal C"), display(C);




% =========================================================================

end
