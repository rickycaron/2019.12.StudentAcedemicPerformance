clear;
dir;
data = readtable('student-mat.csv');%read the file
X=data(:,27:28);% X is the alcohol consumption
grad=data(:,31:33);
y = mean(grad{:,1:end},2); %avgG is the average grade
%y.avgG=round(y.avgG,2);

X=table2array(X);
m=size(X,1);%number o training examples
X = [ones(m,1) X];

[X_norm, mu, sigma] = featureNormalize(X);
%%
alpha = 0.1;
num_iters = 400;
thetanumber=size(X,2);
theta = zeros(thetanumber,1);
[theta,J_history] = gradientDescentMulti(X,y,theta,alpha,num_iters);

% Display gradient descent's result
fprintf('Theta computed from Math gradient descent:\n%f\n%f\n%f',theta(1),theta(2),theta(3));
fprintf('The cost of this agorithum is :\n%f',computeCostMulti(X,y,theta));
% Plot the convergence graph
plot(1:num_iters, J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

