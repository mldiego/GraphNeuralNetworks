%% Verification of all GNN for node classification (qm7) with Linf perturbation

% verify multiple models trained on different random seeds
% Same data for all
% 4 different size of Linf attacks


%% Load data

rng(0); % ensure we can reproduce (data partition)

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
[~ , ~, idxTest] = trainingPartitions(numObservations,[0.8 0.1 0.1]);

% get data from test partition
adjacencyDataTest = adjacencyData(:,:,idxTest);
coulombDataTest = coulombData(:,:,idxTest);
atomDataTest = atomData(idxTest,:);

%% Verify models

% Study Variables
seeds = [0,1,2,3,4]; % models
epsilon = [0.005, 0.01, 0.02, 0.05]; % attack

% Verify one model at a time
for k=1:length(seeds)

    % get model
    modelPath = "models/gcn_"+string(seeds(k))+".mat";

    % Verify model
    verify_model_Linf(modelPath, epsilon, adjacencyDataTest, coulombDataTest, atomDataTest);

end