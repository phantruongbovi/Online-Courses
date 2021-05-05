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
a = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
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
C_min = 0;
sigma_min = 0;
minE = +inf;
for i=1:length(a),
  for j=1:length(a),
    model= svmTrain(X, y, a(i), @(Xval, yval) gaussianKernel(Xval, yval, a(j))); 
    predictions = svmPredict(model, Xval);
    errorP =  mean(double(predictions ~= yval));
    if errorP < minE,
      C_min = a(i);
      sigma_min = a(j);
      minE = errorP;
    endif
  endfor
endfor
C = C_min;
sigma = sigma_min;



% =========================================================================

end
