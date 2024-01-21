%% We will use this one as a general script for verifying GNNs

% what are we doing in this script?
% For every model, there is a set of possible adversarial perturbations
% that we will evaluate. Verification workflow:
% 1) create input sets
% 2) sampling-based falsification search (counterexamples)
% 3) for those not violated, compute reach set
% 4) verify property from computed output sets
% 5) save robustness results