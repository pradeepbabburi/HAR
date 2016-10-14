function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
                                   
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification. 


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));  
Theta2_grad = zeros(size(Theta2));  

% Add bias to the X data matrix
X = [ones(m, 1) X];


% Part 1: Feedforward the neural network and compute the cost
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad.
%
% The vector y passed into the function is a vector of labels
% containing values from 1..K. Need to map this vector into a 
% binary vector to be used with the neural network cost function.
%
% Part 3: Implement regularization with the cost function and gradients.
%
y_matrix = eye(num_labels)(y,:);  % expand y into a true/false matrix(single valued) for all examples
a_1 = X;                          % activation vector - first layer 
a_2 = sigmoid(a_1 * Theta1');     % activation vector  - hidden/second layer 
a_2 = [ones(m, 1) a_2];           % add bias to the vector in previous step
a_3 = sigmoid(a_2 * Theta2');     % activation vector - third/output layer

% Cost function without regularization for the network
cost = -(1/m)*((log(a_3).*y_matrix) + (log(1-a_3).*(1-y_matrix)));    
J = sum(cost(:));
% Regularization term
regz = (lambda/(2*m)) * (sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2))); 
% Add regularization to the cost
J = J + regz;
% Error values, delta for each layer from the right in the network
delta_3 = a_3 - y_matrix;     
delta_2 = delta_3 * Theta2(:, 2:end) .* sigmoidGradient(a_1 * Theta1'); 
% Accumulate delta values for each layer in the network
D_1 = delta_2' * a_1;   
D_2 = delta_3' * a_2;   
% Regularized Gradients
Theta1(:,1) = 0;    %omit the first column from regularization
Theta2(:,1) = 0;
Theta1_grad = (1/m) * (D_1 + (lambda * Theta1));
Theta2_grad = (1/m) * (D_2 + (lambda * Theta2));

% -------------------------------------------------------------

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
