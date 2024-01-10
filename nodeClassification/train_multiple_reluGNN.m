%% Train multiple models to evaluate and certify accuracy and certified accuracy
% For now, do 5 different random seeds and save the model to analyze later

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

rng(2024); % set fix random seed for data

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

seeds = [0,1,2,3,4];

for i=1:length(seeds)
    
    % Set fix random seed for reproducibility
    seed = seeds(i);
    rng(seed);

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
    
    validationFrequency = 100;
    
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
    
    epoch = 0; %initialize epoch
    best_val = 0;
    best_params = [];
    
    t = tic;
    % Begin training (custom train loop)
    while epoch < numEpochs
        epoch = epoch + 1;
    
        % Evaluate the model loss and gradients.
        [loss,gradients] = dlfeval(@modelLoss,parameters,XTrain,ATrain,TTrain);
    
        % Update the network parameters using the Adam optimizer.
        [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients, ...
            trailingAvg,trailingAvgSq,epoch,learnRate);

        % Get validation data
        YValidation = model(parameters,XValidation,AValidation); % output inference
        Yclass = onehotdecode(YValidation,classes,2); % convert to onehot vector
        accVal = mean(Yclass == labelsValidation); % compute accuracy over all validation data

        % update best model
        if accVal > best_val
            best_val = accVal;
            best_params = parameters;
        end
    
        % Display the validation metrics.
        if epoch == 1 || mod(epoch,validationFrequency) == 0
            lossValidation = crossentropy(YValidation,TValidation,DataFormat="BC");
            disp("Epoch = "+string(epoch));
            disp("Loss validation = "+string(lossValidation));
            disp("Accuracy validation = "+string(accVal));
            toc(t);
            disp('--------------------------------------');
        end
    
    end
    
    % save best model
    parameters = best_params;
    
    
    %% Testing
    
    [ATest,XTest,labelsTest] = preprocessData(adjacencyDataTest,coulombDataTest,atomDataTest);
    XTest = (XTest - muX)./sqrt(sigsqX);
    XTest = dlarray(XTest);
    
    YTest = model(parameters,XTest,ATest);
    YTest = onehotdecode(YTest,classes,2);
    
    accuracy = mean(YTest == labelsTest);
    disp("Test accuracy = "+string(accuracy));
    
    % Visualize test results
    figure
    cm = confusionchart(labelsTest,YTest, ...
        ColumnSummary="column-normalized", ...
        RowSummary="row-normalized");
    title("GCN QM7 Confusion Chart");
    
    % Save model
    save("models/gcn_"+string(seed)+".mat", "accuracy", "parameters", "muX", "sigsqX", "best_val");

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
    Z2 = relu(Z2) + Z1;
    % Z2 = relu(Z2);
    
    Z3 = ANorm * Z2 * parameters.mult2.Weights;
    Z3 = relu(Z3) + Z2;
    % Z3 = relu(Z3);
    
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

