%% Create visualizations for computed L_inf results

% We are interested in:
% 1) How many complete molecules are completely robustly verified (all atoms in a moolecule)?
% 2) How many atoms are robustly verified?

%% Process results for each model independently

seeds = [0,1,2,3,4]; % models
epsilon = [0.005; 0.01; 0.02; 0.05];
eN = length(epsilon);

% Verify one model at a time
for m=1:length(seeds)

    % get model
    modelPath = "gcn_"+string(seeds(m));
    
    % initialize vars
    molecules = zeros(eN,4); % # robust, #unknown, # not robust/misclassified, # molecules
    atoms = zeros(eN,4);     % # robust, #unknown, # not robust/misclassified, # atoms
    
    for k = 1:eN
        
        % Load data one at a time
        load("results/verified_nodes_"+modelPath+"_eps"+string(epsilon(k))+".mat");
    
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
    save("results/summary_results_Linf_"+modelPath+".mat", "atoms", "molecules");

    model = load("models/"+modelPath+".mat");
    
    % Create table with these values
    % no need for the molecules, as we could not fully verify any of them
    fileID = fopen("results/summay_results_Linf_"+modelPath+".txt",'w');
    fprintf(fileID, 'Summary of robustness results of gnn model with accuracy = %.4f \n\n', model.accuracy);
    fprintf(fileID, '               MOLECULES \n');
    fprintf(fileID, 'Epsilon | Robust  Unknown  Not Rob.  N \n');
    fprintf(fileID, '  0.005 | %.3f    %.3f   %.3f   %d \n', molecules(1,1)/molecules(1,4), molecules(1,2)/molecules(1,4), molecules(1,3)/molecules(1,4), molecules(1,4));
    fprintf(fileID, '   0.01 | %.3f    %.3f   %.3f   %d \n', molecules(2,1)/molecules(2,4), molecules(2,2)/molecules(2,4), molecules(2,3)/molecules(2,4), molecules(2,4));
    fprintf(fileID, '   0.02 | %.3f    %.3f   %.3f   %d \n', molecules(3,1)/molecules(3,4), molecules(3,2)/molecules(3,4), molecules(3,3)/molecules(3,4), molecules(3,4));
    fprintf(fileID, '   0.05 | %.3f    %.3f   %.3f   %d \n', molecules(4,1)/molecules(4,4), molecules(4,2)/molecules(4,4), molecules(4,3)/molecules(4,4), molecules(4,4));
    fprintf(fileID, ' \n\n');
    fprintf(fileID,'                 ATOMS \n');
    fprintf(fileID, 'Epsilon | Robust  Unknown  Not Rob.  N \n');
    fprintf(fileID, '  0.005 | %.3f    %.3f   %.3f   %d \n', atoms(1,1)/atoms(1,4), atoms(1,2)/atoms(1,4), atoms(1,3)/atoms(1,4), atoms(1,4));
    fprintf(fileID, '   0.01 | %.3f    %.3f   %.3f   %d \n', atoms(2,1)/atoms(2,4), atoms(2,2)/atoms(2,4), atoms(2,3)/atoms(2,4), atoms(2,4));
    fprintf(fileID, '   0.02 | %.3f    %.3f   %.3f   %d \n', atoms(3,1)/atoms(3,4), atoms(3,2)/atoms(3,4), atoms(3,3)/atoms(3,4), atoms(3,4));
    fprintf(fileID, '   0.05 | %.3f    %.3f   %.3f   %d \n', atoms(4,1)/atoms(4,4), atoms(4,2)/atoms(4,4), atoms(4,3)/atoms(4,4), atoms(4,4));
    fclose(fileID);

end