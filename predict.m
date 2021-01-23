function p = predict(theta, X)
  %PREDICT Predict whether the label is 0 or 1 using learned logistic 
  %regression parameters theta
  %   p = PREDICT(theta, X) computes the predictions for X using a 
  %   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
  
  m = size(X, 1); % Number of training examples
  
 
  
 
  % Dimentions:
  % X     =  m x (n+1)
  % theta = (n+1) x 1
  
  h_x = sigmoid(X*theta);
  p=(h_x>=0.5);
  
  %p = double(sigmoid(X * theta)>=0.5);
  % =========================================================================
end
