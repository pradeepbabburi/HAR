function W = randInitializeWeights(L_in, L_out)

%%  RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 

%% Randomly initialize the weights to small values

W = zeros(L_out, 1 + L_in);
epsilon_init = 0.12;
W = rand(L_out, 1+L_in) * 2 * epsilon_init - epsilon_init;

% =========================================================================

end
