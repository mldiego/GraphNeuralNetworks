%% Training example using GCNs
% https://www.mathworks.com/help/deeplearning/ug/node-classification-using-graph-convolutional-network.html

% What is actually happening?
%%%%% DATA
% MATLAB descriptions:
% Molecular data set consisting of 7165 molecules composed of up to 23 atoms. 
% That is, the molecule with the highest number of atoms has 23 atoms.
%
% Official website:
% The dataset is composed of three multidimensional arrays X (7165 x 23 x 23), 
% T (7165) and P (5 x 1433) representing the inputs (Coulomb matrices), 
% the labels (atomization energies) and the splits for cross-validation, respectively. 
% The dataset also contain two additional multidimensional 
% arrays Z (7165) and R (7165 x 3) representing the atomic charge and 
% the cartesian coordinate of each atom in the molecules.
%
% From Pytorch library: 
% https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM7b.html#torch_geometric.datasets.QM7b
% The QM7b dataset from the “MoleculeNet: A Benchmark for Molecular Machine Learning” paper, 
% consisting of 7,211 molecules with 14 regression targets.
%
% STATS:
% #graphs   #nodes     #edges      #features      #tasks
% 7,211     ~15.4      ~245.0          0            14
% 
%%%%% DATA Transformations
% coulomb2Adjacency -> convert the [coulomb,atom] Coulomb matrices to an adjacency matrix
%    adjacency-> [23, 23, 7165],   coulombData-> [23, 23, 7165],   atomData-> [7165, 23]
%
%
% The deep learning model takes as input an adjacency matrix A and a 
% feature matrix X and outputs categorical predictions.
% What are the categorical predictions?
%  - The model predict the atom symbol {'H','C','N','O','S'} of the input
% So now, what is the input?
% - The model takes one input molecule, and the output is a prediction
% probability that correspond to a onehotencode vector 
% There are 5 possible atom scores, in order: {'H','C','N','O','S'}


%% Download data and preprocess it

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

% Visualize data
figure
tiledlayout("flow")

for i = 1:9
    % Extract unpadded adjacency matrix.
    atomicNumbers = nonzeros(atomData(i,:));
    numNodes = numel(atomicNumbers);
    A = adjacencyData(1:numNodes,1:numNodes,i);

    % Convert adjacency matrix to graph.
    G = graph(A);

    % Convert atomic numbers to symbols.
    symbols = atomicSymbol(atomicNumbers);

    % Plot graph.
    nexttile
    plot(G,NodeLabel=symbols,Layout="force")
    title("Molecule " + i)
end

figure
histogram(categorical(atomicSymbol(atomData)))
xlabel("Node Label")
ylabel("Frequency")
title("Label Counts")

% Partition data
numObservations = size(adjacencyData,3);
[idxTrain,idxValidation,idxTest] = trainingPartitions(numObservations,[0.8 0.1 0.1]);

adjacencyDataTrain = adjacencyData(:,:,idxTrain);
adjacencyDataValidation = adjacencyData(:,:,idxValidation);
adjacencyDataTest = adjacencyData(:,:,idxTest);

coulombDataTrain = coulombData(:,:,idxTrain);
coulombDataValidation = coulombData(:,:,idxValidation);
coulombDataTest = coulombData(:,:,idxTest);

atomDataTrain = atomData(idxTrain,:);
atomDataValidation = atomData(idxValidation,:);
atomDataTest = atomData(idxTest,:);

% convert data for training
[ATrain,XTrain,labelsTrain] = preprocessData(adjacencyDataTrain,coulombDataTrain,atomDataTrain);
[AValidation,XValidation,labelsValidation] = preprocessData(adjacencyDataValidation,coulombDataValidation,atomDataValidation);

% normalize training data
muX = mean(XTrain);
sigsqX = var(XTrain,1);

XTrain = (XTrain - muX)./sqrt(sigsqX);
XValidation = (XValidation - muX)./sqrt(sigsqX);


%% Create neural network model

% Initialize models
parameters = struct;

% Layer 1
numHiddenFeatureMaps = 32;
numInputFeatures = size(XTrain,2);

sz = [numInputFeatures numHiddenFeatureMaps];
numOut = numHiddenFeatureMaps;
numIn = numInputFeatures;
parameters.mult1.Weights = initializeGlorot(sz,numOut,numIn,"double");

% Layer 2
sz = [numHiddenFeatureMaps numHiddenFeatureMaps];
numOut = numHiddenFeatureMaps;
numIn = numHiddenFeatureMaps;
parameters.mult2.Weights = initializeGlorot(sz,numOut,numIn,"double");

% Layer 3
classes = categories(labelsTrain);
numClasses = numel(classes);

sz = [numHiddenFeatureMaps numClasses];
numOut = numClasses;
numIn = numHiddenFeatureMaps;
parameters.mult3.Weights = initializeGlorot(sz,numOut,numIn,"double");


%% Training

numEpochs = 1500;
learnRate = 0.01;

validationFrequency = 300;

% initialize params for adam
trailingAvg = [];
trailingAvgSq = [];

% convert data to dlarray for training
XTrain = dlarray(XTrain);
XValidation = dlarray(XValidation);

% gpu?
if canUseGPU
    XTrain = gpuArray(XTrain);
end

% convert labels to onehot vector encoding
TTrain = onehotencode(labelsTrain,2,ClassNames=classes);
TValidation = onehotencode(labelsValidation,2,ClassNames=classes);

% Visualize training progress
monitor = trainingProgressMonitor( ...
    Metrics=["TrainingLoss","ValidationLoss"], ...
    Info="Epoch", ...
    XLabel="Epoch");

groupSubPlot(monitor,"Loss",["TrainingLoss","ValidationLoss"])


epoch = 0; %initialize epoch

% Begin training (custom train loop)
while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;

    % Evaluate the model loss and gradients.
    [loss,gradients] = dlfeval(@modelLoss,parameters,XTrain,ATrain,TTrain);

    % Update the network parameters using the Adam optimizer.
    [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients, ...
        trailingAvg,trailingAvgSq,epoch,learnRate);

    % Record the training loss and epoch.
    recordMetrics(monitor,epoch,TrainingLoss=loss);
    updateInfo(monitor,Epoch=(epoch+" of "+numEpochs));

    % Display the validation metrics.
    if epoch == 1 || mod(epoch,validationFrequency) == 0
        YValidation = model(parameters,XValidation,AValidation);
        lossValidation = crossentropy(YValidation,TValidation,DataFormat="BC");

        % Record the validation loss.
        recordMetrics(monitor,epoch,ValidationLoss=lossValidation);
    end

    monitor.Progress = 100*(epoch/numEpochs);
end


%% Testing

[ATest,XTest,labelsTest] = preprocessData(adjacencyDataTest,coulombDataTest,atomDataTest);
XTest = (XTest - muX)./sqrt(sigsqX);
XTest = dlarray(XTest);

YTest = model(parameters,XTest,ATest);
YTest = onehotdecode(YTest,classes,2);

accuracy = mean(YTest == labelsTest);

% Visualize test results
figure
cm = confusionchart(labelsTest,YTest, ...
    ColumnSummary="column-normalized", ...
    RowSummary="row-normalized");
title("GCN QM7 Confusion Chart");

save('models/gcn.mat', "accuracy", "parameters", "muX", "sigsqX");


%% Predict using new data

numObservationsNew = 4;
adjacencyDataNew = adjacencyDataTest(:,:,1:numObservationsNew);
coulombDataNew = coulombDataTest(:,:,1:numObservationsNew);

predictions = modelPredictions(parameters,coulombDataNew,adjacencyDataNew,muX,sigsqX,classes);

figure
tiledlayout("flow")

for i = 1:numObservationsNew
    % Extract unpadded adjacency data.
    numNodes = find(any(adjacencyDataTest(:,:,i)),1,"last");

    A = adjacencyDataTest(1:numNodes,1:numNodes,i);

    % Create and plot graph representation.
    nexttile
    G = graph(A);
    plot(G,NodeLabel=string(predictions{i}),Layout="force")
    title("Observation " + i + " Prediction")
end




%% Helper functions %%
%%%%%%%%%%%%%%%%%%%%%%

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

function Y = model(parameters,X,A)

    ANorm = normalizeAdjacency(A);
    
    Z1 = X;
    
    Z2 = ANorm * Z1 * parameters.mult1.Weights;
    % Z2 = relu(Z2) + Z1;
    Z2 = relu(Z2);
    
    Z3 = ANorm * Z2 * parameters.mult2.Weights;
    % Z3 = relu(Z3) + Z2;
    Z3 = relu(Z3);
    
    Z4 = ANorm * Z3 * parameters.mult3.Weights;
    Y = softmax(Z4,DataFormat="BC");

end

function [loss,gradients] = modelLoss(parameters,X,A,T)

    Y = model(parameters,X,A);
    loss = crossentropy(Y,T,DataFormat="BC");
    gradients = dlgradient(loss, parameters);

end

function predictions = modelPredictions(parameters,coulombData,adjacencyData,mu,sigsq,classes)

    predictions = {};
    numObservations = size(coulombData,3);
    
    for i = 1:numObservations
        % Extract unpadded data.
        numNodes = find(any(adjacencyData(:,:,i)),1,"last");
        A = adjacencyData(1:numNodes,1:numNodes,i);
        X = coulombData(1:numNodes,1:numNodes,i);
    
        % Preprocess data.
        [A,X] = preprocessPredictors(A,X);
        X = (X - mu)./sqrt(sigsq);
        X = dlarray(X);
    
        % Make predictions.
        Y = model(parameters,X,A);
        Y = onehotdecode(Y,classes,2);
        predictions{end+1} = Y;
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

