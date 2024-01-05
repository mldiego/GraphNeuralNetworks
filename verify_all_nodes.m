% Verification of a Graph Neural Network

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

%% Start for loop for verification here, preprocess one molecule at a time

N = size(coulombDataTest, 3);
% Get data statistics from training data (need to get this from
% training,statistics are approx, get the exact one after retraining)
muX = mean(XTrain);
sigsqX = var(XTrain,1);

% Store resuts
targets = {};
outputSets = {};
rT = {};

for i = 1:N

    % preprocess test data (717 molecules in test set)
    % [ATest,XTest,labelsTest] = preprocessData(adjacencyDataTest,coulombDataTest,atomDataTest);
    % verify just one molecule?
    [ATest,XTest,labelsTest] = preprocessData(adjacencyDataTest(:,:,N),coulombDataTest(:,:,N),atomDataTest(N,:));
    
    % normalize data
    XTest = (XTest - muX)./sqrt(sigsqX);
    XTest = dlarray(XTest);

    % Create an input set
    
    % adjacency matrix represent connections, so keep it as is
    Averify = normalizeAdjacency(ATest);
    
    % input values for each node is X
    lb = extractdata(XTest-0.01);
    ub = extractdata(XTest+0.01);
    Xverify = ImageStar(lb,ub);
    
    % Compute reachability
    t = tic;
    
    reachMethod = 'approx-star';
    L = ReluLayer(); % Create relu layer;
    
    Y = computeReachability({w1,w2,w3}, L, reachMethod, Xverify, Averify);
    outputSets{i} = Y;
    targets{i} = labelsTest;
    
    % % Get output bounds
    % [yLower, yUpper] = Y.getRanges();
    rT{i} = toc(t);

end


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

function Y = computeReachability(weights, L, reachMethod, input, adjMat)
    % weights = weights of GNN ({w1, w2, w3}
    % L = Layer type (ReLU)
    % reachMethod = reachability method for all layers('approx-star is default)
    % input = pertubed input features (ImageStar)
    % adjMat = adjacency matric of corresonding input features
    % Y = computed output of GNN (ImageStar)

    Xverify = input;
    Averify = adjMat;
    n = size(adjMat,1);
    
    %%%%%%%%  LAYER 1  %%%%%%%%
    
    % part 1
    newV = Xverify.V;
    newV = reshape(newV, [n n+1]);
    newV = Averify * newV;
    newV = tensorprod(newV, extractdata(weights{1}));
    newV = permute(newV, [1 4 3 2]);
    X2 = ImageStar(newV, Xverify.C, Xverify.d, Xverify.pred_lb, Xverify.pred_ub);
    % part 2
    X2b = L.reach(X2, reachMethod);
    repV = repmat(Xverify.V,[1,32,1,1]);
    Xrep = ImageStar(repV, Xverify.C, Xverify.d, Xverify.pred_lb, Xverify.pred_ub);
    X2b_ = X2b.MinkowskiSum(Xrep);
    
    %%%%%%%%  LAYER 2  %%%%%%%%
    
    % part 1
    newV = X2b_.V;
    newV = tensorprod(full(Averify), newV, 2, 1);
    newV = tensorprod(newV, extractdata(weights{2}),2,1);
    newV = permute(newV, [1 4 2 3]);
    X3 = ImageStar(newV, X2b_.C, X2b_.d, X2b_.pred_lb, X2b_.pred_ub);
    % part 2
    X3b = L.reach(X3, reachMethod); 
    X3b_ = X3b.MinkowskiSum(X2b_);
    
    %%%%%%%%  LAYER 3  %%%%%%%%
    
    newV = X3b_.V;
    newV = tensorprod(full(Averify), newV, 2, 1);
    newV = tensorprod(newV, extractdata(weights{3}), 2, 1);
    newV = permute(newV, [1 4 2 3]);
    Y = ImageStar(newV, X3b_.C, X3b_.d, X3b_.pred_lb, X3b_.pred_ub);

end

