data = readmatrix("heart.csv"); % import csv file to data variable.
data = data(randperm(size(data, 1)), :);% our data is in order 1 to 0. we must shuffle
% our data matrix. otherwise if we dont do that our test examples will be
% include like %80 1's and %20 0's even its %50 %50 in our original
% example. if we dont shuffle our data , we can't predict new patients
% right.
x = data(1:240,1:13); % taking %80 of our data for train case.
x_test = data(241:303,1:13); % we will use last %20 example for test case.
y = data(1:240,14);% train outputs
y_test = data(241:303,14);% test outputs.
[m, n] = size(x); % m = number of training example. , n = number of total feature.
x = [ones(m, 1) x]; % intercept term. ( including bias term to each example as first column.)
initial_theta = zeros(n + 1, 1); % Initialize the fitting parameters. feature+1 because we added column of 1's.
lambda = 0.3; % our regularization parameter is 0.3. ( i tried some options and 0.3 is the best one.)
[cost, grad] = costFunctionReg(initial_theta, x, y,lambda); % we are looking our default cost function.
% aim is minimizing this cost function. grad is our derivative
% values. we will use them in advanced cost function named "fminunc" .
options = optimoptions(@fminunc,'Algorithm','Quasi-Newton','GradObj', 'on', 'MaxIter', 80);
% we did some set up before finding parameter thetas. we mentioned we will
% use grad,maximum iteration is 80.
[theta, cost] = fminunc(@(t)(costFunctionReg(t, x, y,lambda)), initial_theta, options);
% we sent our default thetas to find some better thetas. after thetas are
% calculated, we found our new cost. this new one is much lower than we did
% before.
p = predict(theta, x); % predicting our train data with parameter thetas.
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
% Train Accuracy: should be something between %80-%90.
% as you remember , we shuffle our data matrix before. since we shuffle our
% data everytime randomly, we couldnt know exact accuracy. but i have tried
% more than hundred times and its always around %85. 
% our main aim is not getting good score in train data. because we trained
% our parameters for these examples. of course we will get good training
% score. our aim is continuing this success in test data.
x_test = [ ones(size(x_test,1),1) x_test ];
p = predict(theta, x_test); % predicting test data accuracy.
fprintf('Test Accuracy: %f\n', mean(double(p == y_test)) * 100);
%Test Accuracy: should be something between 80-90.
%there is no problem such as high bias,high variance problems with our training parameters.
%our test accuracy value is so close to training accuracy. as i mentioned
%because we shuffle our data , we cant know exact number but as i tried
%hundred times, it is around 84. rarely sometimes down to 80 and up to 90.
%since we got right parameters. we can try new patients if they have heart
%disease or not.

% if we wanna know if a patient has heart disease or not. after putting 1
% into first column. we can predict his/her values with our trained thetas.

% example :
deneme = [ 1 41 1 3 135 165 0 1 178 0 0 2 3 2 ]; % lets say this is the values
% of a suspicious patient. as we compare these values with our examples ,
% its high likely not heart disease values. but lets see what we predict.
sonuc = predict(theta,deneme) % we send our thetas with example values.
%deneme =

%  logical

%   1

%we get logical 1. our suspicious patient is healthy as we expected.
