%% Running a Simple GNN example from networkx and numpy
% https://towardsdatascience.com/understanding-graph-convolutional-networks-for-node-classification-a2bfdb7aba7b
% This is a simple example to understand how GNNs work in general
% There is no trainig, just like a toy example

%% Initialize the Graph G (input to neural network)
% The graph G will consist of 6 nodes and the feature of each node 
% will correspond to that particular node number.
% 
% Graph looks like this:
%
%  4               1
%   \             /
%    \           /
%     3 - - - - 0
%    /           \
%   /             \
%  5               2
%

% Adjacency matrix (How the nodes are connected)
A = [0 1 1 1 0 0;
    1 0 1 0 0 0;
    1 1 0 0 0 0;
    1 0 0 0 1 1;
    0 0 0 1 0 1;
    0 0 0 1 1 0];
% Node features matrix (values of node)
X = [0 1 2 3 4 5]';

% Then, we want to do some "sum of neighboring features"
% A*X is not good enough
% So we add self-loops and normalization
I = eye(length(X));
A = A + I; % Self-loop

AX = A*X;

% Two ways to normalize the features (D^-1AX) and (D^-(1/2)AXD^-(1/2))
% Second one seems to work better, or so they claim. 
% For our analyses, it would depend on how we train

D = [5 0 0 0 0 0;
    0 4 0 0 0 0;
    0 0 4 0 0 0;
    0 0 0 5 0 0;
    0 0 0 0 4 0;
    0 0 0 0 0 4];

% D_inv = inv(D);
D_inv = D^-1;
D_inv_half = D^-(1/2);

% Normalizing adjacency matrix (AX)
DAX = D_inv*AX;
DADX = D_inv_half*A*D_inv_half*X;

% Input is ready now (DADX), build a simple GNN

%% Create GNN (weights and activation function)
% This is a 2-layer network with relu (poslin) activation function

rng(77777);
n_h = 4; %number of neurons in the hidden layer
n_y = 2; %number of neurons in the output layer

W0 = randn(size(X,2),n_h)*0.01;
W1 = randn(n_h, n_y)*0.01;

% Build GCN Layer
% function y = gcn(A,X,W)
%     I = eye(length(A)); % identity matrix
%     A_ = A+I; % add self-loops
%     D = diag(sum(A_,1)); %degree matrix
%     D_inv_half = D^-(1/2);
%     y = D_inv_half*A*D_inv_half*X*W; % input * weights
%     y = poslin(y); % apply activation function
% end

% Compute the output of each layer (Forward prop)
H1 = gcn(A,X,W0);
H2 = gcn(A,H1,W1);

%% Plotting Feature Representation

x = H2(:,1);
y = H2(:,2);

figure;
sz = 100;
scatter(x,y,sz);
xlim([min(x)*0.9, max(x)*1.1]);
ylim([-1 1]);


%% Helper Functions 
function y = gcn(A,X,W)
    I = eye(length(A)); % identity matrix
    A_ = A+I; % add self-loops
    D = diag(sum(A_,1)); %degree matrix
    D_inv_half = D^-(1/2);
    y = D_inv_half*A*D_inv_half*X*W; % input * weights
    y = poslin(y); % apply activation function
end



%% Notes
% Looking at how the inputs are typically constructed, it may make sense to 
% add uncertainty to X (node features), but should leave edges how they are.
% I don't think we should be taking the inverse of uncertainty matrices,
% not sure that will work, uncertainty will probably be very large with the
% tinites perturbation. Does it make sense to "attack" GNNs by modifying
% edges?

% Idea, add some perturbation to X, then compute the reachability analysis.
% Key: create a "GraphStar" set in NNV, which contains A and X, which A
% never changes, but keeps getting propagates through the layers
% Should be similar to an ImageStar (grayscale) with an extra value A

% In terms of the network, we can have it as a CNN in NNV, as this can be
% applied to be a general function. 