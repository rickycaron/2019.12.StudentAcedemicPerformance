%%
clear;
dir;
%read the file
data = readtable('student-por.csv');
%choose which columns will be considered from data
inputdatacolumn=(1:30);
%inputdatacolumn=[3,7,8,13:30];
%input all the chosen columns into x from data
X=data(:,inputdatacolumn);
% transfer all the boolean column into 0 and 1
X.schoolsup = (categorical(X.schoolsup) == 'yes');
X.famsup = (categorical(X.famsup) == 'yes');
X.paid = (categorical(X.paid) == 'yes');
X.activities = (categorical(X.activities) == 'yes');
X.nursery = (categorical(X.nursery) == 'yes');
X.higher =  (categorical(X.higher) == 'yes');
X.internet = (categorical(X.internet) == 'yes');
X.romantic = (categorical(X.romantic) == 'yes');
X.school = (categorical(X.school) == 'GP');
X.sex = (categorical(X.sex) == 'F');
X.address = (categorical(X.address) == 'U');
X.famsize = (categorical(X.famsize) == 'LE3');
X.Pstatus = (categorical(X.Pstatus) == 'T');
X.Mjob = grp2idx(categorical(X.Mjob));
X.Fjob = grp2idx(categorical(X.Fjob));
X.reason = grp2idx(categorical(X.reason));
X.guardian = grp2idx(categorical(X.guardian));
%transfer table to array
%%
origintable=X;
propertyname=(origintable.Properties.VariableNames)';
X=table2array(origintable);
%all the grades of the students
grad=data(:,31:33);
y = mean(grad{:,1:end},2); %avgG is the average grade
y=round(y,0);
%%
 [m,n]=size(X);%number o training examples
%add the first column of const 1
X = [ones(m,1) X];
%%
X_norm=X;
for i = 2:size(X,2)
    [X_norm(:,i), mu, sigma] = featureNormalize(X(:,i));
end
% Randomly select X
sel = randperm(m);
Xtrain=X_norm(sel(1:390),:);
ytrain=y(sel(1:390),:);
Xval=X_norm(sel(391:520),:);
yval=y(sel(391:520),:);
Xtest=X_norm(sel(521:end),:);
ytest=y(sel(521:end),:);
%%
%linear regression
alpha = 0.1;
num_iters = 500;
thetanumber=size(Xtrain,2);
theta = zeros(thetanumber,1);
[theta,J_history] = gradientDescentMulti(Xtrain,ytrain,theta,alpha,num_iters);
% Display gradient descent's result
fprintf('The cost of this agorithum of Potgue is :%f',computeCostMulti(Xtrain,ytrain,theta));
% Plot the convergence graph
plot(1:num_iters, J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
theta;
fprintf('\nLinear regression Cross-validation Test Accuracy: %f\n', mean(double(abs(Xtest*theta-ytest)<= 1)) * 100);
%%
% normal equation
theta = pinv(Xg'*Xg)*Xg'*y;
computeCostMulti(Xg,y,theta);
theta;
%% Neutral Network
%lambdaarray=[0,0.1,0.2,0.4,0.8,1.6,3.2,6.4];

input_layer_size  = size(Xtrain,2);  % input layer
hidden_layer_size = 50;   % 
num_labels = 20;          % 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)

% Weight regularization parameter (we set this to 0 here).
% generate epsilon and theta
epsilon_init1 = sqrt(6)/sqrt(input_layer_size+hidden_layer_size);
epsilon_init2 = sqrt(6)/sqrt(hidden_layer_size+num_labels);
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size,epsilon_init1);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels,epsilon_init2);
% Unroll parameters
nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
%  Check gradients by running checkNNGradients
lambda = 8;
checkNNGradients(lambda); 
debug_J  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
fprintf('Cost at (fixed) debugging parameters (w/ lambda = 3): %f', debug_J);
options = optimset('MaxIter', 1000);
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, Xtrain, ytrain, lambda);
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, ~] = fmincg(costFunction, nn_params, options);
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
 
pred = predict(Theta1, Theta2, Xtrain);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == ytrain)) * 100);
predval = predict(Theta1, Theta2, Xval);
fprintf('\nCross-validation Test Accuracy: %f\n', mean(double(abs(predval- yval)<= 1)) * 100);
predtest = predict(Theta1, Theta2, Xtest);
fprintf('\nTest Set Accuracy: %f\n', mean(double(abs(predtest- ytest)<= 1)) * 100);
fprintf('\nTest Set Accuracy: %f\n', sum((predtest- ytest).^2)/size(ytest,1);
%%
%linear regression
lambda = 0;
[theta] = trainLinearReg(Xtrain, ytrain, lambda);
[error_train, error_val] = learningCurve(Xtrain, ytrain, Xval, yval, lambda);

plot(1:size(error_train,1), error_train, 1:size(error_train,1), error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
%%
p = 3;
lambda = 10;
% Map X onto Polynomial Features and Normalize
%X_poly now is the column of 1's
X_poly = polyFeatures(Xtrain(:,2:end), p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(size(X_poly, 1), 1), X_poly];  % Add Ones
% % Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest(:,2:end), p);
X_poly_test = X_poly_test-mu; % uses implicit expansion instead of bsxfun
X_poly_test = X_poly_test./sigma; % uses implicit expansion instead of bsxfun
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones
% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval(:,2:end), p);
X_poly_val = X_poly_val-mu; % uses implicit expansion instead of bsxfun
X_poly_val = X_poly_val./sigma; % uses implicit expansion instead of bsxfun
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

[theta] = trainLinearReg(X_poly, ytrain, lambda);
%[error_train, error_val] = learningCurve(X_poly, ytrain, X_poly_val, yval, lambda);
% plot(1:size(error_train,1), error_train, 1:size(error_val,1), error_val);
% title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
% xlabel('Number of training examples')
% ylabel('Error')
% legend('Train', 'Cross Validation')
%%
pol=6;
[lambda_vec, error_train, error_val] = validationCurve(Xtrain, ytrain, Xval, yval,pol);
%%
min_val_error= min(min(error_val));
[i,j]=find(error_val==min_val_error);
%%

lambda = 10;
p = 3;
% Map X onto Polynomial Features and Normalize
%X_poly now is the column of 1's
X_poly = polyFeatures(Xtrain(:,2:end), p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(size(X_poly, 1), 1), X_poly];  % Add Ones

% % Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest(:,2:end), p);
X_poly_test = X_poly_test-mu; % uses implicit expansion instead of bsxfun
X_poly_test = X_poly_test./sigma; % uses implicit expansion instead of bsxfun
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones
%%
% [theta] = trainLinearReg(X_poly, ytrain, lambda);
% [error_test] = linearRegCostFunction(X_poly_test, ytest, theta, 0) 
predtestlr=X_poly_test*theta;
fprintf('\nTest Set Accuracy 0: %f\n', mean(double(predtestlr== ytest)) * 100);
fprintf('\nTest Set Accuracy rang1: %f\n', mean(double(abs(predtestlr- ytest)<= 1)) * 100);


