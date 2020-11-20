function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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

combos = [0.01 0.03 0.1 0.3 1 3 10 30];
n = length(combos);

% Matrix to hold results of using (c_ind, sig_ind) for cross_validation. 
pred_errors = zeros(n, n);

for c_ind = 1:length(combos)
	for sig_ind = 1:length(combos)
		C_train = combos(c_ind);
		sig_train = combos(sig_ind);
	
		model = svmTrain(X, y, C_train, @(x1, x2) gaussianKernel(x1, x2, sig_train));
		
		% get predictions for current choice of params and compute prediction error	
		predictions = svmPredict(model, Xval);
		pred_errors(c_ind, sig_ind) = mean(double(predictions ~= yval));
	endfor
endfor

% Find C and sigma value where cross validation error is minimum
min_error = min(min(pred_errors));
[min_i min_j] = find(pred_errors == min_error);

C = combos(min_i);
sigma = combos(min_j);
 

% =========================================================================

end
