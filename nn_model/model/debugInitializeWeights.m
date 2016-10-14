function W = debugInitializeWeights(n_out, n_in)

%   W = DEBUGINITIALIZEWEIGHTS(n_in, n_out) initializes the weights 
%   of a layer with n_in incoming connections and n_out outgoing 
%   connections using a fix set of values
%
%   Note that W should be set to a matrix of size(1 + n_in, n_out) as
%   the first row of W handles the "bias" term
%

% Set W to zeros
W = zeros(n_out, 1 + n_in);

% Initialize W using "sin", this ensures that W is always of the same
% values and will be useful for debugging
W = reshape(sin(1:numel(W)), size(W)) / 10;

% =========================================================================

end
