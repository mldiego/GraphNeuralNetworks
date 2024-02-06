%% Create visualizations for computed L_inf results

% We are interested in:
% 1) How many complete molecules are completely robustly verified (all atoms in a moolecule)?
% 2) How many atoms are robustly verified?

%% Process results for each model independently

% variables
seeds = 0:1:9; % models
epsilon = [0.005; 0.01; 0.02; 0.05];
nPixels = 10:10:100;
reachMethod = "approx-star";
% how many adversarial robustness to consider? (eN x nS x nP)
eN = length(epsilon);
nS = length(seeds);
nP = length(nPixels);

% Verify one model at a time
for m=1:length(seeds)

    % get model
    modelPath = "gcn_"+string(seeds(m));

    for k = 1:eN
    
        % initialize vars
        molecules = zeros(nP,4); % # robust, #unknown, # not robust/misclassified, # molecules
        atoms = zeros(nP,4);     % # robust, #unknown, # not robust/misclassified, # atoms
        
        for p=1:length(nPixels)
            
            % Load data one at a time
            load("resultsNEW/verified_nodes_"+modelPath+"_"+reachMethod+"_eps"+string(epsilon(k))+"_"+string(nPixels(p))+".mat");
        
            N = length(targets);
            
            for i=1:N
                
                % get result data
                res = results{i};
                n = length(res);
                rb  = sum(res==1); % robust
                unk = sum(res==2); % unknown
                nrb = sum(res==0); % not robust
                
                % molecules
                if rb == n
                    molecules(k,1) = molecules(k,1) + 1;
                elseif unk == n
                    molecules(k,2) = molecules(k,2) + 1;
                elseif nrb == n
                    molecules(k,3) = molecules(k,3) + 1;
                end
                molecules(k,4) = molecules(k,4) + 1;
                
                % atoms
                atoms(k,1) = atoms(k,1) + rb;
                atoms(k,2) = atoms(k,2) + unk;
                atoms(k,3) = atoms(k,3) + nrb;
                atoms(k,4) = atoms(k,4) + n;
        
            end
                
        end

    % Save summary
    save("resultsNEW/summary_results_Linf_" + modelPath + "_pixels_" + string(epsilon(k)) + ".mat", "atoms", "molecules");

    model = load("models/"+modelPath+".mat");
    
    % Create table with these values
    % no need for the molecules, as we could not fully verify any of them
    fileID = fopen("resultsNEW/summary_results_Linf_" + modelPath + "_pixels_" + string(epsilon(k)) +".txt",'w');
    % Show the overall accuracy
    fprintf(fileID, 'Summary of robustness results of gnn model with accuracy = %.4f \n\n', model.accuracy);
    % Molecules
    fprintf(fileID, '               MOLECULES \n');
    fprintf(fileID, 'Pixels (%%) | Robust  Unknown  Not Rob.  N \n');
    for p=1:nP
        fprintf(fileID, '   %d       | %.3f    %.3f   %.3f   %d \n', nPixels(p), molecules(p,1)/molecules(p,4), molecules(p,2)/molecules(p,4), molecules(p,3)/molecules(p,4), molecules(p,4));
    end
    fprintf(fileID, ' \n\n');
    % Atoms
    fprintf(fileID,'                 ATOMS \n');
    fprintf(fileID, 'Pixels (%%) | Robust  Unknown  Not Rob.  N \n');
    for p=1:nP
        fprintf(fileID, '   %d       | %.3f    %.3f   %.3f   %d \n', nPixels(p), atoms(p,1)/atoms(p,4), atoms(p,2)/atoms(p,4), atoms(p,3)/atoms(p,4), atoms(p,4));
    end
    fclose(fileID);
        
    end

end

