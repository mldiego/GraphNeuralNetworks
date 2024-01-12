%% Verify the robustness reach sets of all models

epsilon = [0.005; 0.01; 0.02; 0.05];
seeds = [0,1,2,3,4];

for m=1:length(seeds)

    % get model
    modelPath = "gcn_"+string(seeds(m));
    
    for k = 1:length(epsilon)
    
        % Load data one at a time
        load("results/verified_nodes_"+modelPath+"_eps"+string(epsilon(k))+".mat");
    
        % Check for robustness value (one molecule, 1 atom at a time)
        results = {};
        for i=1:length(outputSets)
            Y = outputSets{i};
            label = targets{i};
            results{i} = verifyAtom(Y, label);
        end
    
        % Save results
        save("results/verified_nodes_"+modelPath+"_eps"+string(epsilon(k))+".mat", "results", "outputSets", "rT", "targets");
        
    end

end


function results = verifyAtom(X, target)
    % Check verification result (robustness) for every atom
    Natom = size(target,1);
    results = 3*ones(Natom,1); % initialize results array
    for i=1:Natom
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

