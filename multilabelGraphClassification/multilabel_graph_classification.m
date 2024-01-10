%% Multilabel Graph Classification Using Graph Attention Networks

doTraining = true;

%% Data
% https://zenodo.org/records/4288677

% download data
zipFile = matlab.internal.examples.downloadSupportFile("nnet","data/QM7X.zip");
dataFolder = fileparts(zipFile);
unzip(zipFile,dataFolder);

load(fullfile(dataFolder,"QM7X","QM7X.mat"))

% Extract and concatenate the node features.
features = cat(3,dataQM7X.hDIP,dataQM7X.atPOL,dataQM7X.vdwR);
features = permute(features,[1 3 2]);

% Extract the atomic numbers and the coordinates data and use them to build the adjacency matrix
atomicNumber = dataQM7X.atNUM;
coordinates = dataQM7X.atXYZ;
adjacency = coordinates2Adjacency(coordinates,atomicNumber);
% Extract the labels 
labels = uniqueFunctionalGroups(adjacency,atomicNumber);

% Partition the data into training, validation, and test partitions 
% containing 80%, 10%, and 10% of the data, respectively
numGraphs = size(adjacency,3);
[idxTrain,idxValidation,idxTest] = trainingPartitions(numGraphs,[0.8 0.1 0.1]);

featuresTrain = features(:,:,idxTrain);
featuresValidation = features(:,:,idxValidation);
featuresTest = features(:,:,idxTest);

adjacencyTrain = adjacency(:,:,idxTrain);
adjacencyValidation = adjacency(:,:,idxValidation);
adjacencyTest = adjacency(:,:,idxTest);

labelsTrain = labels(idxTrain);
labelsValidation = labels(idxValidation);
labelsTest = labels(idxTest);

% Normalize the features using the mean and variance of the nonzero elements of the training features
numFeatures = size(featuresTrain,2);
muX = zeros(1,numFeatures);
sigsqX = zeros(1,numFeatures);

for i = 1:numFeatures
    X = nonzeros(featuresTrain(:,i,:));
    muX(i) = mean(X);
    sigsqX(i) = var(X, 1);
end

% Normalize the training features, excluding the zero elements that pad the data.
numGraphsTrain = size(featuresTrain,3);

for j = 1:numGraphsTrain
    validIdx = 1:nnz(featuresTrain(:,1,j));
    featuresTrain(validIdx,:,j) = (featuresTrain(validIdx,:,j) - muX)./sqrt(sigsqX);
end

% Normalize the validation features using the same statistics and also 
% exclude the zero elements that pad the data.
numGraphsValidation = size(featuresValidation,3);
for j = 1:numGraphsValidation
    validIdx = 1:nnz(featuresValidation(:,1,j));
    featuresValidation(validIdx,:,j) = (featuresValidation(validIdx,:,j) - muX)./sqrt(sigsqX);
end

% Get the class names from the label data
classNames = unique(cat(1,labels{:}));

% Encode the training labels into a binary array of size numObservations-by-numClasses, 
% where numObservations is the number of observations and numClasses is the number of classes. 
% In each row, the nonzero entries correspond to the labels of each observation.
TTrain = zeros(numGraphsTrain,numel(classNames));

for j = 1:numGraphsTrain
    if ~isempty(labelsTrain{j})
        [~,idx] = ismember(labelsTrain{j},classNames);
        TTrain(j,idx) = 1;
    end
end

% Visualize the number of graphs per class using a bar graph.
classCounts = sum(TTrain,1);

figure
bar(classCounts)
ylabel("Count")
xticklabels(classNames)

% Visualize the number of labels per graph using a histogram
labelCounts = sum(TTrain,2);

figure
histogram(labelCounts)
xlabel("Number of Labels")
ylabel("Frequency")

% Encode the validation labels into a binary array
TValidation = zeros(numGraphsValidation,numel(classNames));
for j = 1:numGraphsValidation
    if ~isempty(labelsValidation{j})
        [~,idx] = ismember(labelsValidation{j},classNames);
        TValidation(j,idx) = 1;
    end
end

% To train using mini-batches of data, create array datastores for 
% the feature, adjacency, and target training data and combine them.
featuresTrain = arrayDatastore(featuresTrain,IterationDimension=3);
adjacencyTrain = arrayDatastore(adjacencyTrain,IterationDimension=3);
targetTrain = arrayDatastore(TTrain);

dsTrain = combine(featuresTrain,adjacencyTrain,targetTrain);

% To make predictions using mini-batches of data, create an array datastore 
% for the validation features and adjacency data and combine them.
featuresValidation = arrayDatastore(featuresValidation,IterationDimension=3);
adjacencyValidation = arrayDatastore(adjacencyValidation,IterationDimension=3);
dsValidation = combine(featuresValidation,adjacencyValidation);


%% Graph model
% The model takes as input a feature matrix X and an adjacency matrix A and outputs categorical predictions.
% The model uses a masked multihead self attention mechanism to aggregate 
% features across the neighborhood of a node, that is, the set of nodes  that are 
% directly connected to the node. The mask, which is obtained from the adjacency matrix, 
% is used to prevent attention between nodes that are not in the same neighborhood.

% Specify 3 heads for the first and second attention operations. 
% Specify 5 heads for the third attention operation.
numHeads = struct;
numHeads.attn1 = 3;
numHeads.attn2 = 3;
numHeads.attn3 = 5;

% Initialize the weights of the first attention operation to have an output size of 96. 
% The input size is the number of channels of the input feature data.
parameters = struct;
numInputFeatures = size(features,2);
numHiddenFeatureMaps = 96;
numClasses = numel(classNames);

sz = [numInputFeatures numHiddenFeatureMaps];
numOut = numHiddenFeatureMaps;
numIn = numInputFeatures;

parameters.attn1.weights.linearWeights = initializeGlorot(sz,numOut,numIn);
parameters.attn1.weights.attentionWeights = initializeGlorot([numOut 2],1,2*numOut);

% Initialize the weights of the second attention operation to have the same 
% output size as the previous multiply operation. 
% The input size is the output size of the previous attention operation.

sz = [numHiddenFeatureMaps numHiddenFeatureMaps];
numOut = numHiddenFeatureMaps;
numIn = numHiddenFeatureMaps;

parameters.attn2.weights.linearWeights = initializeGlorot(sz,numOut,numIn);
parameters.attn2.weights.attentionWeights = initializeGlorot([numOut 2],1,2*numOut);

% Initialize the weights of the third attention operation to have an output size 
% matching the number of classes. The input size is the output size of the previous attention operation
numOutputFeatureMaps = numHeads.attn3*numClasses;

sz = [numHiddenFeatureMaps numOutputFeatureMaps];
numOut = numClasses;
numIn = numHiddenFeatureMaps;
parameters.attn3.weights.linearWeights = initializeGlorot(sz,numOut,numIn);
parameters.attn3.weights.attentionWeights = initializeGlorot([numOutputFeatureMaps 2],1,2*numOut);



%% Training

% Train for 70 epochs with a mini-batch size of 300. 
% Large mini-batches of training data for GATs can cause out-of-memory errors.
numEpochs = 70;
miniBatchSize = 300;

learnRate = 0.01;
labelThreshold = 0.5;
validationFrequency = 210;

% Use minibatchqueue to process and manage mini-batches of training data
mbq = minibatchqueue(dsTrain,4, ...
    MiniBatchSize=miniBatchSize, ...
    PartialMiniBatch="discard", ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    OutputCast="double", ...
    OutputAsDlarray=[1 0 0 0], ...
    OutputEnvironment = ["auto" "cpu" "cpu" "cpu"]);

% The function makes predictions by iterating over mini-batches of data 
% using the read size property of the datastore object. Set the read size 
% properties of the array datastore holding the validation data to miniBatchSize. 
dsValidation.UnderlyingDatastores{1}.ReadSize = miniBatchSize;
dsValidation.UnderlyingDatastores{2}.ReadSize = miniBatchSize;

% Initialize the parameters for Adam optimizer
trailingAvg = [];
trailingAvgSq = [];

if doTraining

    % Initialize the training progress plot.
    figure
    C = colororder;
    
    lineLossTrain = animatedline(Color=C(2,:));
    lineLossValidation = animatedline( ...
        LineStyle="--", ...
        Marker="o", ...
        MarkerFaceColor="black");
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Loss")
    grid on

    iteration = 0;
    start = tic;
    
    % Loop over epochs.
    for epoch = 1:numEpochs

        % Shuffle data.
        shuffle(mbq);
            
        while hasdata(mbq)
            iteration = iteration + 1;
            
            % Read mini-batches of data.
            [XTrain,ATrain,numNodes,TTrain] = next(mbq);
    
            % Evaluate the model loss and gradients using dlfeval and the
            % modelLoss function.
            [loss,gradients,Y] = dlfeval(@modelLoss,parameters,XTrain,ATrain,numNodes,TTrain,numHeads);
            
            % Update the network parameters using the Adam optimizer.
            [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients, ...
                trailingAvg,trailingAvgSq,iteration,learnRate);
            
            % Display the training progress.
            D = duration(0,0,toc(start),Format="hh:mm:ss");
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            loss = double(loss);
            addpoints(lineLossTrain,iteration,loss)
            drawnow
    
            % Display validation metrics.
            if iteration == 1 || mod(iteration,validationFrequency) == 0
                YValidation = modelPredictions(parameters,dsValidation,numHeads);
                lossValidation = crossentropy(YValidation,TValidation,ClassificationMode="multilabel",DataFormat="BC");

                lossValidation = double(lossValidation);
                addpoints(lineLossValidation,iteration,lossValidation)
                drawnow
            end
        end
    end
else
    % Load the pretrained parameters (provided by MathWorks)
    load("parametersQM7X_GAT.mat")
end


%% Test/Evaluate

% Preferable to use GPU for training
if canUseGPU
    featuresTest = gpuArray(featuresTest);
end

% Normalize the test features using the statistics of the training features
numGraphsTest = size(featuresTest,3);

for j = 1:numGraphsTest
    validIdx = 1:nnz(featuresTest(:,1,j));
    featuresTest(validIdx,:,j) = (featuresTest(validIdx,:,j) - muX)./sqrt(sigsqX);
end

% Create array datastores for the test features and adjacency data, 
% setting their ReadSize properties to miniBatchSize, and combine them.
featuresTest = arrayDatastore(featuresTest,IterationDimension=3,ReadSize=miniBatchSize);
adjacencyTest = arrayDatastore(adjacencyTest,IterationDimension=3,ReadSize=miniBatchSize);
dsTest = combine(featuresTest,adjacencyTest);

% Encode the test labels into a binary array.
TTest = zeros(numGraphsTest,numel(classNames));

for j = 1:numGraphsTest
    if ~isempty(labelsTest{j})
        [~,idx] = ismember(labelsTest{j},classNames);
        TTest(j,idx) = 1;
    end
end

% Predict and Convert prediction probabilities to binary encoded labels using a 
% label threshold of 0.5, which is the same as the label threshold labelThreshold, 
% used when training and validating the model.
predictions = modelPredictions(parameters,dsTest,numHeads);
predictions = double(gather(extractdata(predictions)));
YTest = double(predictions >= 0.5);

% Evaluate the performance by calculating the F-score using the fScore function, 
% defined in the F-Score Function section of the example.
%
% The fScore function uses a weighting parameter beta to place greater value on either precision or recall. 
% Precision is the ratio of true positive results to all positive results, 
% including those that are incorrectly predicted as positive
% 
% Recall is the ratio of true positive results to all actual positive samples.
%
% Calculate the F-score using three weighting parameters:
%    0.5 — Precision is twice as important as recall.
%      1 — Precision and recall are equally important. Use this value to monitor the performance of the model during training and validation.
%      2 — Recall is twice as important as precision.

scoreWeight = [0.5 1 2];
for i = 1:3
    scoreTest(i) = fScore(YTest,TTest,scoreWeight(i));
end

% View the scores in a table.
scoreTestTbl = table;
scoreTestTbl.Beta = scoreWeight';
scoreTestTbl.FScore = scoreTest';

% Visualize the confusion chart for each class
figure
tiledlayout("flow")
for i = 1:numClasses
    nexttile
    confusionchart(YTest(:,i),TTest(:,i));
    title(classNames(i))
end

% Visualize the receiver operating characteristics (ROC) curves for each class.
% -> The ROC curve plots the true positive rates versus false positive rates and 
%    illustrates the performance of the model at all labeling thresholds. 
% -> The true positive rate is the ratio of true positive results to all actual positive samples, 
%    including those that the model incorrectly predicts as negative. 
% -> The false positive rate is the ratio of false positive results to all actual negative samples, 
%    including those that are incorrectly predicted as positive.
%
% The area under the curve (AUC) provides an aggregate measure of performance across all possible labeling thresholds.
% For each class:
%    Compute the true positive rates and false positive rates using the roc function.
%    Calculate the AUC using the trapz function.
%    Plot the ROC curve and display the AUC. Also plot the ROC curve of a random, or 
%     no-skill, model that makes random predictions, or always predicts the same result.
figure
tiledlayout("flow")

for i = 1:numClasses
    currentTargets = TTest(:,i)';
    currentPredictions = predictions(:,i)';

    [truePositiveRates,falsePositiveRates] = roc(currentTargets,currentPredictions);
    AUC = trapz(falsePositiveRates,truePositiveRates);

    nexttile
    plot(falsePositiveRates,truePositiveRates, ...
        falsePositiveRates,falsePositiveRates,"--",LineWidth=0.7)
    text(0.075,0.75,"\bf AUC = "+num2str(AUC),FontSize=6.75)
    xlabel("FPR")
    ylabel("TPR")
    title(classNames(i))
end

lgd = legend("ROC Curve - GAT", "ROC Curve - Random");
lgd.Layout.Tile = numClasses+1;

save('gat_relu.mat', 'parameters', 'AUC', 'scoreTestTbl', 'muX', 'sigsqX')


%% Predict using new data

% Load the preprocessed QM7X sample data
load(fullfile(dataFolder,"QM7X","preprocessedQM7XSample.mat"))
% Get the adjacency matrix and node features from the sample data.
adjacencyMatrixSample = dataSample.AdjacencyMatrix;
featuresSample = dataSample.Features;
% View the number of nodes in the graph.
numNodesSample = size(adjacencyMatrixSample,1);

% Extract the graph data. To compute the attention scores, remove added self-connections 
% from the adjacency matrix, then use the matrix to construct the graph.
A = adjacencyMatrixSample - eye(numNodesSample);
G = graph(A);

% Map the atomic numbers to symbols
atomicNumbersSample = dataSample.AtomicNumbers;
[symbols,symbolsCount] = atomicSymbol(atomicNumbersSample);

% Display the graph in a plot, using the mapped symbols as node labels.
figure
plot(G,NodeLabel=symbols,LineWidth= 0.75,Layout="force")
title("Sample Molecule")

XSample = dlarray(featuresSample);
if canUseGPU
    XSample = gpuArray(XSample);
end

% Make predictions using the model function. Also obtain the attention scores computed by the attention operator in the model
[YSample,attentionScores] = model(parameters,XSample,adjacencyMatrixSample,numNodesSample,numHeads);

% Convert prediction probabilities to binary encoded labels
YSample = gather(extractdata(YSample));
YSample = YSample >= 0.5;

% Convert the predicted binary labels to actual labels
predictionsSample = classNames(YSample);

% Visualize the attention scores.
% Create a heat map of the attention scores per head in the final attention operation of the model using heatmap.
attention3Scores = double(gather(extractdata(attentionScores.attn3)));
numHeadsAttention3 = numHeads.attn3;

figure
tiledlayout("flow")
for i = 1:numHeadsAttention3
    nexttile
    heatmap(symbolsCount,symbolsCount,attention3Scores(:,:,i),ColorScaling="scaledrows",Title="Head "+num2str(i))
end

% The x and y values in the attention maps correspond to the node labels in the plot below.
figure
plot(G,NodeLabel=symbolsCount,LineWidth= 0.75,Layout="force")




%% Helper functions

% The model function takes as inputs the model parameters parameters, 
% the feature matrix X, the adjacency matrix A, the number of nodes per graph numNodes, 
% and the number of heads numHeads, and returns the predictions and the attention scores.
function [Y,attentionScores] = model(parameters,X,A,numNodes,numHeads)

    weights = parameters.attn1.weights;
    numHeadsAttention1 = numHeads.attn1;
    
    Z1 = X;
    [Z2,attentionScores.attn1] = graphAttention(Z1,A,weights,numHeadsAttention1,"cat");
    Z2  = relu(Z2);
    
    weights = parameters.attn2.weights;
    numHeadsAttention2 = numHeads.attn2;
    
    [Z3,attentionScores.attn2] = graphAttention(Z2,A,weights,numHeadsAttention2,"cat");
    % Z3  = relu(Z3) + Z2;
    Z3  = relu(Z3) + Z2;
    
    weights = parameters.attn3.weights;
    numHeadsAttention3 = numHeads.attn3;
    
    [Z4,attentionScores.attn3] = graphAttention(Z3,A,weights,numHeadsAttention3,"mean");
    Z4 = globalAveragePool(Z4,numNodes);
    
    Y = sigmoid(Z4);

end

% Returns the gradients of the loss with respect to the model parameters, the corresponding loss, and the model predictions
function [loss,gradients,Y] = modelLoss(parameters,X,adjacencyTrain,numNodes,T,numHeads)
    Y = model(parameters,X,adjacencyTrain,numNodes,numHeads);
    loss = crossentropy(Y,T,ClassificationMode="multilabel",DataFormat="BC");
    gradients = dlgradient(loss,parameters);
end

% merges mini-batches of different graph instances into a single graph instance
function [features,adjacency,numNodes,target] = preprocessMiniBatch(featureData,adjacencyData,targetData)

    % Extract feature and adjacency data from their cell array and concatenate the
    % data along the third (batch) dimension
    featureData = cat(3,featureData{:});
    adjacencyData = cat(3,adjacencyData{:});
    
    % Extract target data if it exists
    if nargin > 2
        target = cat(1,targetData{:});
    end
    
    adjacency = sparse([]);
    features = [];
    numNodes = [];
    
    for i = 1:size(adjacencyData, 3)
        % Get the number of nodes in the current graph
        numNodesInGraph = nnz(featureData(:,1,i));
        numNodes = [numNodes; numNodesInGraph];
    
        % Get the indices of the actual nonzero data
        validIdx = 1:numNodesInGraph;
    
        % Remove zero paddings from adjacencyData
        tmpAdjacency = adjacencyData(validIdx, validIdx, i);
    
        % Add self connections
        tmpAdjacency = tmpAdjacency + eye(size(tmpAdjacency));
    
        % Build the adjacency matrix into a block diagonal matrix
        adjacency = blkdiag(adjacency, tmpAdjacency);
    
        % Remove zero paddings from featureData
        tmpFeatures = featureData(validIdx, :, i);
        features = [features; tmpFeatures];
    end

end

% The fScore function calculates the micro-average F-score, which measures 
% the model accuracy on the data using the precision and the recall.
function score = fScore(predictions,targets,beta)

    truePositive = sum(predictions .* targets,"all");
    falsePositive = sum(predictions .* (1-targets),"all");
    
    % Precision
    precision = truePositive/(truePositive + falsePositive);
    
    % Recall
    recall = truePositive/sum(targets,"all");
    
    % FScore
    if nargin == 2
        beta = 1;
    end
    
    score = (1+beta^2)*precision*recall/(beta^2*precision+recall);

end

% Computes the model predictions by iterating over mini-batches of data and 
% preprocessing each mini-batch using the preprocessMiniBatch function.
function Y = modelPredictions(parameters,ds,numHeads)

    Y = [];
    
    reset(ds)
    
    while hasdata(ds)
    
        data = read(ds);
    
        featureData = data(:,1);
        adjacencyData = data(:,2);
    
        [features,adjacency,numNodes] = preprocessMiniBatch(featureData,adjacencyData);
    
        X = dlarray(features);
    
        minibatchPred = model(parameters,X,adjacency,numNodes,numHeads);
        Y = [Y;minibatchPred];
    end

end

% The graphAttention function computes node features using masked multihead self-attention.
function [outputFeatures,normAttentionCoeff] = graphAttention(inputFeatures,adjacency,weights,numHeads,aggregation)

    % Split weights with respect to the number of heads and reshape the matrix to a 3-D array
    szFeatureMaps = size(weights.linearWeights);
    numOutputFeatureMapsPerHead = szFeatureMaps(2)/numHeads;
    linearWeights = reshape(weights.linearWeights,[szFeatureMaps(1), numOutputFeatureMapsPerHead, numHeads]);
    attentionWeights = reshape(weights.attentionWeights,[numOutputFeatureMapsPerHead, 2, numHeads]);
    
    % Compute linear transformations of input features
    value = pagemtimes(inputFeatures,linearWeights);
    
    % Compute attention coefficients
    query = pagemtimes(value, attentionWeights(:, 1, :));
    key = pagemtimes(value, attentionWeights(:, 2, :));
    
    attentionCoefficients = query + permute(key,[2, 1, 3]);
    attentionCoefficients = leakyrelu(attentionCoefficients,0.2);
    
    % Compute masked attention coefficients
    mask = -10e9 * (1 - adjacency);
    attentionCoefficients = attentionCoefficients + mask;
    
    % Compute normalized masked attention coefficients
    normAttentionCoeff = softmax(attentionCoefficients,DataFormat = "BCU");
    
    % Normalize features using normalized masked attention coefficients
    headOutputFeatures = pagemtimes(normAttentionCoeff,value);
    
    % Aggregate features from multiple heads
    if strcmp(aggregation, "cat")
        outputFeatures = headOutputFeatures(:,:);
    else
        outputFeatures =  mean(headOutputFeatures,3);
    end

end

% The elu function implements the ELU activation
function y = elu(x)

    y = max(0, x) + (exp(min(0, x)) -1);

end

% Computes an output feature representation for each graph by averaging 
% the input features with respect to the number of nodes per graph.
function outFeatures = globalAveragePool(inFeatures,numNodes)

    numGraphs = numel(numNodes);
    numFeatures = size(inFeatures, 2);
    outFeatures = zeros(numGraphs,numFeatures,"like",inFeatures);
    
    startIdx = 1;
    for i = 1:numGraphs
        endIdx = startIdx + numNodes(i) - 1;
        idx = startIdx:endIdx;
        outFeatures(i,:) = mean(inFeatures(idx,:));
        startIdx = endIdx + 1;
    end

end


