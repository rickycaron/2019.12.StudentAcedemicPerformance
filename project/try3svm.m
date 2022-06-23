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
%%
y=double((y >=10));
%%
 [m,n]=size(X);%number o training examples
%add the first column of const 1
%%
% Randomly select X
sel = randperm(m);
Xtrain=X(sel(1:390),:);
ytrain=y(sel(1:390),:);
Xval=X(sel(391:520),:);
yval=y(sel(391:520),:);
Xtest=X(sel(521:end),:);
ytest=y(sel(521:end),:);
%%
%svm
[C, sigma,errors] = dataset3Params(Xtrain, ytrain, Xval, yval);
%%
% SVM Parameters
C = 0.1;
sigma=5;
% We set the tolerance and max_passes lower here so that the code will run faster. However, in practice, 
% you will want to run the training to convergence.
model= svmTrain(Xtrain, ytrain, C, @(x1, x2)gaussianKernel(x1, x2, sigma)); 
%model = svmTrain(X, y, C, @linearKernel);
%model = svmTrain(Xtrain, ytrain, C, @linearKernel);
predictions = svmPredict(model, Xval);
mean(double(predictions == yval))*100;
%%
predictions = svmPredict(model, Xtest);
mean(double(predictions == ytest))*100


