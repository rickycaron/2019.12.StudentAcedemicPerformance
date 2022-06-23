function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval,pol)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%
% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30]';
%lambda_vec = [0 0.001 0.003]';
% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), pol);
error_val = zeros(length(lambda_vec), pol);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%
%       for i = 1:length(lambda_vec)
%           lambda = lambda_vec(i);
%           % Compute train / val errors when training linear 
%           % regression with regularization parameter lambda
%           % You should store the result in error_train(i)
%           % and error_val(i)
%           ....
%           
%       end
%
%

for p = 1:pol
    X_poly = polyFeatures(X(:,2:end), p);
    [X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
    X_poly = [ones(size(X_poly, 1), 1), X_poly];  % Add Ones

    % Map X_poly_val and normalize (using mu and sigma)
    X_poly_val = polyFeatures(Xval(:,2:end), p);
    X_poly_val = X_poly_val-mu; % uses implicit expansion instead of bsxfun
    X_poly_val = X_poly_val./sigma; % uses implicit expansion instead of bsxfun
    X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

    for i = 1:length(lambda_vec)
       lambda = lambda_vec(i);
       [theta] = trainLinearReg(X_poly, y, lambda);
       [error_train(i,p), grad] = linearRegCostFunction(X_poly, y, theta, 0);
       [error_val(i,p), grad] = linearRegCostFunction(X_poly_val, yval, theta, 0);           
    end
end










% =========================================================================

end
