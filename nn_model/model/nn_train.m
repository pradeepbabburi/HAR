
%%===== Building and Training the classification model using a three layer Neural Network =====%%

%% Initialization
clear ; close all; clc

%% Setup the parameters
input_layer_size  = 561;  % number of features
hidden_layer_size = 30;   % hidden layer size
num_labels = 6;          % number of labels - output layer size   

%% Load Training Data
fprintf('Loading Training Data ...\n')

load('samsungData_train');
m = size(X_train, 1);

%% Randomly select 20 data points to display
sel = randperm(size(X_train, 1));
sel = sel(1:20);
X_train(sel, 1:10)
%y_train(sel)

fprintf('Program paused. Press enter to continue.\n')
pause;

%% normalize the features
fprintf('Normalizing features ...\n')
mu = mean(X_train);
sigma = std(X_train);
X_train = (X_train - mu) ./ sigma;

fprintf('Showing normalized data ..\n')
X_train(sel,1:10)
fprintf('Program paused. Press enter to continue.\n')
pause;

%% Initialize network parameters
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

%% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% Training the neural network
%
fprintf('\nTraining Neural Network... \n')

%% set some hyper parameters
options = optimset('MaxIter', 100);
lambda = 1;

%% Create short hand for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

%% Get Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Training complete. Program paused. Press enter to continue.\n');
pause;


%% Predict the model with trained Theta values

pred = predict(Theta1, Theta2, X_train);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_train)) * 100);
fprintf('Program paused. Press enter to continue.\n');
pause;

%% predict the model on test data
fprintf('Loading Test Data ...\n')
load('samsungData_test');
X_test = (X_test - mean(X_test))./std(X_test);
pred_test = predict(Theta1, Theta2, X_test);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred_test == y_test)) * 100);



