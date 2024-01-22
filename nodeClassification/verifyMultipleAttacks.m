%% We will use this one as a general script for verifying GNNs

% what are we doing in this script?
% For every model, there is a set of possible adversarial perturbations
% that we will evaluate. Verification workflow:
% 1) create input sets
% 2) sampling-based falsification search (counterexamples)
% 3) for those not violated, compute reach set
% 4) verify property from computed output sets
% 5) save robustness results

% study variables (Linf perturbation on feature input vector X)
seeds = 0:9;
xPerc = 10:10:100; % percentage of X features perturbed
epsilon = [0.005, 0.01, 0.02, 0.05]; % attack
nSamples = 100; % try 100 samples for falsification
reachMethod = 'approx-star';
L = ReluLayer(); % Create relu layer (part of the model, common for all of them)
classes = {'H';'C';'N';'O';'S'};

% Other attacks?
% Look at single feature attacks? 
% Edge removal?

%% Load the data (same for all attacks and every model)
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


%% Main 
% Look at one model at a time
for k=1:length(seeds)

    % get model
    modelPath = "gcn_"+string(seeds(k));

    % Load parameters of gcn
    net = load("models/"+modelPath+".mat");
    
    w1 = gather(net.parameters.mult1.Weights);
    w2 = gather(net.parameters.mult2.Weights);
    w3 = gather(net.parameters.mult3.Weights);
    muX = net.muX;
    sigsqX = net.sigsqX;

    % Start for loop for verification here, preprocess one molecule at a time
    
    N = size(coulombDataTest, 3);

    rng(0); % reset seed to ensure same features are perturbed across models
    
    for ep = 1:length(epsilon)% size of attack

        EPSILON = epsilon(ep);

        for p = 1:length(xPerc) % percentage of X features perturbed
            
            XPERC = xPerc(p);
            VT = {};
            results = {};

            parfor i = 1:N
                
                %%% Process data to create input set
                % Get molecule data
                [ATest,XTest,labelsTest] = preprocessData(adjacencyDataTest(:,:,N),coulombDataTest(:,:,N),atomDataTest(N,:));
                
                % normalize data
                XTest = (XTest - muX)./sqrt(sigsqX);
                XTest = dlarray(XTest);
                        
                % adjacency matrix represent connections, so keep it as is
                Averify = normalizeAdjacency(ATest);
                
                % 1) Create input set based on attack (epsilon & xPerc)
                % Get input set: input values for each node is X
                nT = length(XTest);
                nXF = ceil(XPERC*nT/100); % number of features perturbed
                XF = randperm(nT,nXF); % perturbed features
                IDXS = zeros(nT,1);
                IDXS(XF) = 1;
                IDXS = EPSILON*IDXS;
                % define bounds
                lb = extractdata(XTest-IDXS);
                ub = extractdata(XTest+IDXS);
                Xverify = ImageStar(lb,ub); % for verification

                % 2) Falsification
                t = tic;
                Xbox = Box(lb,ub); % for falsification
                Xsamples = Xbox.sample(nSamples);
                % Initialize var
                resVER = 2*ones(size(lb)); % unknown to start for all atoms
                % Any atoms missclassified?
                YTest = model(net.parameters,XTest,ATest);
                YTest = onehotdecode(YTest,classes,2);
                predV = YTest == labelsTest;
                missIdxs = find(predV == 0); % missclassified ones
                resVER(missIdxs) = -1;
                % Begin falsification
                for s=1:nSamples
                    YTest = model(net.parameters,Xsamples(:,s),ATest);
                    YTest = onehotdecode(YTest,classes,2);
                    predV = YTest == labelsTest;
                    resVER = resVER .*predV; % change to 0 those falsified
                    if all(resVER < 1) % unless all atoms are falsified (misclassified), keep going
                        break;
                    end
                end
                resVER(missIdxs) = -1; % add the missclasified ones at the end as well in case they were converted to 0
                tF = toc(t);
                
                if ~resVER % counterexample found, no reachability neeeded
                    continue
                end
                
                % 3) Compute reachability
                t = tic;
                Y = computeReachability({w1,w2,w3}, L, reachMethod, Xverify, Averify);
                tR = toc(t);

                % 4) Verify output set
                t = tic;
                resVER = verifyAtom(Y, labelsTest, resVER);
                tV = toc(t);

                % store results
                results{i} = resVER; % save verification result
                VT{i} = [tF;tR;tV]; % total verification time

            end

            % 5) Save results
            parsave(modelPath, EPSILON, XPERC, results, VT, reachMethod);

        end

    end

end


%% Helper functions

function results = verifyAtom(X, target, results)
    % Check verification result (robustness) for every atom
    Natom = size(target,1);
    for i=1:Natom
        if results(i)  < 1 % falsified or misclassifed
            continue
        end
        matIdx = zeros(1,Natom);
        matIdx(i) = 1;
        Y = X.affineMap(matIdx, []); % Reduce Imagestar to 1 dimension
        Y = Y.toStar; % convert to star
        atomLabel = target(i,:);
        atomHs = label2Hs(atomLabel);
        res = verify_specification(Y,atomHs);
        if res == 2
            % check is propery is violated
            res = checkViolated(Y, target(i,:));
        end
        results(i) = res;
    end
end

function res = checkViolated(Set, label)
    res = 2; % assume unknown (property is not unsat, try to sat)
    % get target label index
    switch label
        case 'H'
            target = 1;
        case 'C'
            target = 2;
        case 'N'
            target = 3;
        case 'O'
            target = 4;
        case 'S'
            target = 5;
    end
    % Get bounds for every index
    [lb,ub] = Set.getRanges;
    maxTarget = ub(target);
    % max value of the target index smaller than any other lower bound?
    if any(lb > maxTarget)
        res = 0; % falsified
    end
end

function Hs = label2Hs(label)
    % Convert output target to halfspace for verification
    % @Hs: unsafe/not robust region defined as a HalfSpace

    outSize = 5; % num of classes
    % classes = ["H";"C";"N";"O";"S"];

    switch label
        case 'H'
            target = 1;
        case 'C'
            target = 2;
        case 'N'
            target = 3;
        case 'O'
            target = 4;
        case 'S'
            target = 5;
    end

    % Define HalfSpace Matrix and vector
    G = ones(outSize,1);
    G = diag(G);
    G(target, :) = [];
    G = -G;
    G(:, target) = 1;

    g = zeros(size(G,1),1);

    % Create HalfSapce to define robustness specification
    Hs = [];
    for i=1:length(g)
        Hs = [Hs; HalfSpace(G(i,:), g(i))];
    end

end

function parsave(modelPath, epsilon, features, results, VT, reachMethod)
    save("resultsNEW/verified_nodes_" + modelPath + "_" + reachMethod + "_eps" + string(epsilon)+"_" + string(features) + ".mat", "results", "VT");
end

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

function Y = model(parameters,X,ANorm)

    % ANorm = normalizeAdjacency(A);
    
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
