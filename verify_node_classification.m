% Verification of a Graph Convolutional Neural Network

%% Load parameters of gcn
model = load('gcn.mat');

w1 = gather(model.parameters.mult1.Weights);
w2 = gather(model.parameters.mult2.Weights);
w3 = gather(model.parameters.mult3.Weights);

% model function
%     ANorm => adjacency matrix of A
%     Z1 => input
% 
%     Z2 = ANorm * Z1 * w1;
%     Z2 = relu(Z2) + Z1; (layer 1)
% 
%     Z3 = ANorm * Z2 * w2;
%     Z3 = relu(Z3) + Z2; (layer 2)
% 
%     Z4 = ANorm * Z3 * w3;
%     Y = softmax(Z4,DataFormat="BC"); (output layer)

%% Load data

rng(0); % ensure we can reproduce

dataURL = "http://quantum-machine.org/data/qm7.mat";
outputFolder = fullfile(tempdir,"qm7Data");
dataFile = fullfile(outputFolder,"qm7.mat");

if ~exist(dataFile,"file")
    mkdir(outputFolder);
    disp("Downloading QM7 data...");
    websave(dataFile, dataURL);
    disp("Done.")
end

data = load(dataFile);
% Extract the Coulomb data and the atomic numbers from the loaded structure. 
% Permute the Coulomb data so that the third dimension corresponds to the observations. 
coulombData = double(permute(data.X, [2 3 1]));
% Sort the atomic numbers in descending order.
atomData = sort(data.Z,2,'descend');
% convert data to adjacency form
adjacencyData = coulomb2Adjacency(coulombData,atomData);

% Partition data
numObservations = size(adjacencyData,3);
[idxTrain ,~, idxTest] = trainingPartitions(numObservations,[0.8 0.1 0.1]);

% training data
adjacencyDataTrain = adjacencyData(:,:,idxTrain);
coulombDataTrain = coulombData(:,:,idxTrain);
atomDataTrain = atomData(idxTrain,:);
[ATrain,XTrain,labelsTrain] = preprocessData(adjacencyDataTrain,coulombDataTrain,atomDataTrain);

% get data from test partition
adjacencyDataTest = adjacencyData(:,:,idxTest);
coulombDataTest = coulombData(:,:,idxTest);
atomDataTest = atomData(idxTest,:);

% preprocess test data
% [ATest,XTest,labelsTest] = preprocessData(adjacencyDataTest,coulombDataTest,atomDataTest);
% verify just one molecule?
[ATest,XTest,labelsTest] = preprocessData(adjacencyDataTest(:,:,1),coulombDataTest(:,:,1),atomDataTest(1,:));

% Get data statistics from training data (need to get this from
% training,statistics are approx, get the exact one after retraining)
muX = mean(XTrain);
sigsqX = var(XTrain,1);

% normalize data
XTest = (XTest - muX)./sqrt(sigsqX);
XTest = dlarray(XTest);

% We'll use some examples from the test data to verify


%% Create an input set

% adjacency matrix represent connections, so keep it as is
Averify = normalizeAdjacency(ATest);

% input values for each node is X
lb = extractdata(XTest-0.1);
ub = extractdata(XTest+0.1);
Xverify = ImageStar(lb,ub);

% Do we need a new representation for graphs?


%% Compute reachability
L = ReluLayer(); % Create relu layer;

%%%%%%%%  LAYER 1  %%%%%%%%

% inference with original input
Z2 = Averify * XTest * w1;
Z2_ = relu(Z2) + XTest;

% inference with lower bound
lb2 = Averify * lb * w1;
lb2_ = relu(lb2) + lb;

% inference with upper bound
ub2 = Averify * ub * w1;
ub2_ = relu(ub2) + ub;

% reachability
% for the first step, we only need to work on the basis vectors
newV = Xverify.V;
newV = reshape(newV, [16 17]);
newV = Averify * newV;
newV = tensorprod(newV, extractdata(w1));
newV = permute(newV, [1 4 3 2]);
X2 = ImageStar(newV, Xverify.C, Xverify.d, Xverify.pred_lb, Xverify.pred_ub);
% part 2
X2b = L.reach(X2, 'approx-star');
% for i = size()
% X2 = Xverify.affineMap(Averify, []);
% newV = Averify * Xverify.V(:,1) * w1;
% newV = X2.V * w1;
% X2 = X2.affineMap(w1, []);
% X2 = L.reach(X2, 'exact-star');


%%%%%%%%  LAYER 2  %%%%%%%%
 
% inference with original input
Z3 = Averify * Z2_ * w2;
Z3_ = relu(Z3) + Z2_;

% inference with lower bound
lb3 = Averify * lb2_ * w2;
lb3_ = relu(Z3) + lb2_;

% inference with upper bound
ub3 = Averify * ub2_ * w2;
ub3_ = relu(ub3) + ub2_;

% reachability


%%%%%%%%  LAYER 3  %%%%%%%%
 
% inference with original input
Z4 = Averify * Z3_ * w3;
% Y = softmax(Z4,DataFormat="BC");

% inference with lower bound
lb4 = Averify * lb3_ * w3;
% Y = softmax(Z4,DataFormat="BC");

% inference with upper bound
ub4 = Averify * ub3_ * w3;
% Y = softmax(Z4,DataFormat="BC");

% reachability



%% Notes
% Although simple enough in this example for inference, there seems to be
% several challenges for reachability
% 1) What is the "input set"?
%    - The input is adjacency matrix + X (node values), but there are used
%    differently. Do we need a new data structure to represent these
%    graphs?
% 2) "Bias" in relu layers
%    - What typically is the bias in other NN-types, here is the input to
%    the layer, which has different dimensions than current "set".
%    Minkowski sum should not work, so let's see how we can manage that.


%% Helper functions

function [adjacency,features,labels] = preprocessData(adjacencyData,coulombData,atomData)

    [adjacency, features] = preprocessPredictors(adjacencyData,coulombData);
    labels = [];
    
    % Convert labels to categorical.
    for i = 1:size(adjacencyData,3)
        % Extract and append unpadded data.
        T = nonzeros(atomData(i,:));
        labels = [labels; T];
    end
    
    labels2 = nonzeros(atomData);
    assert(isequal(labels2,labels2))
    
    atomicNumbers = unique(labels);
    atomNames =  atomicSymbol(atomicNumbers);
    labels = categorical(labels, atomicNumbers, atomNames);

end

function [adjacency,features] = preprocessPredictors(adjacencyData,coulombData)

    adjacency = sparse([]);
    features = [];
    
    for i = 1:size(adjacencyData, 3)
        % Extract unpadded data.
        numNodes = find(any(adjacencyData(:,:,i)),1,"last");
    
        A = adjacencyData(1:numNodes,1:numNodes,i);
        X = coulombData(1:numNodes,1:numNodes,i);
    
        % Extract feature vector from diagonal of Coulomb matrix.
        X = diag(X);
    
        % Append extracted data.
        adjacency = blkdiag(adjacency,A);
        features = [features; X];
    end

end

function ANorm = normalizeAdjacency(A)

    % Add self connections to adjacency matrix.
    A = A + speye(size(A));
    
    % Compute inverse square root of degree.
    degree = sum(A, 2);
    degreeInvSqrt = sparse(sqrt(1./degree));
    
    % Normalize adjacency matrix.
    ANorm = diag(degreeInvSqrt) * A * diag(degreeInvSqrt);

end
