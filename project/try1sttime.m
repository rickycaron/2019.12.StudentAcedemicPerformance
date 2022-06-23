clear;
dir;
data = readtable('student-por.csv');%read the file
X=data(:,27:28);% X is the alcohol consumption
grad=data(:,31:33);
y = mean(grad{:,1:end},2); %avgG is the average grade
%y.avgG=round(y.avgG,2);

X=table2array(X);
m=size(X,1);%number o training examples
X = [ones(m,1) X];

%%
alpha = 0.1;
num_iters = 400;
thetanumber=size(X,2);
theta = zeros(thetanumber,1);
[theta,J_history] = gradientDescentMulti(X,y,theta,alpha,num_iters);

% Display gradient descent's result
fprintf('Theta computed from Portugues gradient descent:\n%f\n%f\n%f',theta(1),theta(2),theta(3));
fprintf('The cost of this agorithum of Potgue is :\n%f',computeCostMulti(X,y,theta));
% Plot the convergence graph
plot(1:num_iters, J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');


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


% X.MjobTeacher= (X.Mjob=="teacher");
% X.MjobHealth= (X.Mjob=="health");
% X.MjobServices= (X.Mjob=="services");
% X.MjobAthome= (X.Mjob=="at_home");
% X.MjobOther= (X.Mjob=="other");
% 
% X.FjobTeacher= (X.Fjob=="teacher");
% X.FjobHealth= (X.Fjob=="health");
% X.FjobServices= (X.Fjob=="services");
% X.FjobAthome= (X.Fjob=="at_home");
% X.FjobOther= (X.Fjob=="other");
% 
% X.reasonhome= (X.reason=="home");
% X.reasonreputaion= (X.reason=="reputaion");
% X.reasoncourse= (X.reason=="course");
% X.reasonother= (X.reason=="other");
% 
% X.guardianM = (X.guardian=="mother");
% X.guardianF = (X.guardian=="father");
% X.guardianO = (X.guardian=="other");

X.Mjob = grp2idx(categorical(X.Mjob));
X.Fjob = grp2idx(categorical(X.Fjob));
X.reason = grp2idx(categorical(X.reason));
X.guardian = grp2idx(categorical(X.guardian));
% X.Mjob=[];
% X.Fjob=[];
% X.reason=[];
% X.guardian=[];
%transfer table to array
%%
origintable=X;
propertyname=(X.Properties.VariableNames)'
X=table2array(X);

%all the grades of the students
grad=data(:,31:33);
y = mean(grad{:,1:end},2); %avgG is the average grade
%y.avgG=round(y.avgG,2);

m=size(X,1);%number o training examples
%add the first column of const 1
X = [ones(m,1) X];
%[X(:,2), mu, sigma] = featureNormalize(X(:,2));

%%
X_norm=X;
%for i = [4,8,9,10,11,21,25,26,27]
for i = 2:size(X,2)
    [X_norm(:,i), mu, sigma] = featureNormalize(X(:,i));
end

%%
Xg=X_norm;
alpha = 0.1;
num_iters = 500;
thetanumber=size(Xg,2);
theta = zeros(thetanumber,1);
[theta,J_history] = gradientDescentMulti(Xg,y,theta,alpha,num_iters);

% Display gradient descent's result
fprintf('Theta computed from Portugues gradient descent:\n%f\n%f\n%f',theta(1),theta(2),theta(3));
fprintf('The cost of this agorithum of Potgue is :\n%f',computeCostMulti(Xg,y,theta));
% Plot the convergence graph
plot(1:num_iters, J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
theta
%%

theta = pinv(Xg'*Xg)*Xg'*y;
computeCostMulti(Xg,y,theta);
theta

