function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m,1) X];
hidden_layer_a = [ones(m,1) sigmoid(X * Theta1')];
yh_output = sigmoid(hidden_layer_a * Theta2');
y_matrix = eye(num_labels)(y, :);

J = sum(sum(-y_matrix .* log(yh_output) - (1-y_matrix) .* log(1-yh_output)));
J = sum(J) / m;

t1 = Theta1(:, 2:end);
t2 = Theta2(:, 2:end); % Also used in below backpropagation
reg = sum(sum(t1 .* t1));
reg = reg + sum(sum(t2 .* t2));
reg = reg * lambda / m;

J = J + reg/2;


Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

Z2 = X*Theta1';
A2 = [ones(m, 1) sigmoid(Z2)];
A3 = sigmoid(A2*Theta2');

D3 = (A3 - y_matrix)';
D2 = t2'*D3 .* sigmoidGradient(Z2)';
Delta1 = D2 * X;
Delta2 = D3 * A2;

t1 = lambda*t1/m;
t2 = lambda*t2/m;
t1_reg = [zeros(size(Theta1,1), 1) t1];
t2_reg = [zeros(size(Theta2,1), 1) t2];

Theta1_grad = Delta1/ m + t1_reg;
Theta2_grad = Delta2/ m + t2_reg;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
