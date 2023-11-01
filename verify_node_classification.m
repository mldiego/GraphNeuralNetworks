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
Xverify = Star(lb,ub);


%% Compute reachability
L = ReluLayer(); % Create relu layer;

%  LAYER 1
% inference
Z2 = Averify * XTest * w1;
Z2 = relu(Z2) + XTest;
% reachability
X2 = Xverify.affineMap(Averify, []);
newV = X2.V * w1;
X2 = X2.affineMap(w1, []);
X2 = L.reach(X2, 'exact-star');
 
%  LAYER 2
% inference
Z3 = ANorm * Z2 * w2;
Z3 = relu(Z3) + Z2;
% reachability


%  LAYER 3
% inference
Z4 = ANorm * Z3 * w3;
Y = softmax(Z4,DataFormat="BC");
% reachability




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
