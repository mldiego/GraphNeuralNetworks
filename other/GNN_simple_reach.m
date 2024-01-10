%% Reachability analysis of a simple GNN
% What do we have?

% 1) In terms of Graph NN
% Inputs (graph): X, A
% Model (NN): Two GCN layers with relu activations

% 2) In terms of NNV
% ImageStar: can use this as a set representation
% Poslin reach methods for Star and ImageStar
% Matrix multiplications with ImageStars (affineMap?)

% Time to start, make a prototype, and then we can try to generalize

%% Load GNN and input (using GNN_simple_example.m)

% Adjacency matrix (How the nodes are connected)
A = [0 1 1 1 0 0;
    1 0 1 0 0 0;
    1 1 0 0 0 0;
    1 0 0 0 1 1;
    0 0 0 1 0 1;
    0 0 0 1 1 0];
% Node features matrix (values of node)
X = [0 1 2 3 4 5]';

rng(77777);
n_h = 4; %number of neurons in the hidden layer
n_y = 2; %number of neurons in the output layer

W0 = randn(size(X,2),n_h)*0.01;
W1 = randn(n_h, n_y)*0.01;

%% Start the process

A_ = A + eye(length(X)); % Self-loop
D = diag(sum(A_,1)); %degree matrix
D_inv_half = D^-(1/2);
% DADX = D_inv_half*A*D_inv_half*X;
IS = ImageStar(X,X); % Simplest test is when IB = UB

% Transform input first outside of the layer methods
DAD = D_inv_half*A*D_inv_half;

L = ReluLayer(); % Create relu layer

% Forward Propagation
% Layer 1
IS = IS.affineMap(DAD,[]);
IS = IS.affineMap(W0,[]);
R1 = L.reach(IS,'exact-star');
% Layer 2
Rt = R1.affineMap(DAD,[]);
Rt = Rt.affineMap(W1,[]);
R = L.reach(Rt,'exact-star');






