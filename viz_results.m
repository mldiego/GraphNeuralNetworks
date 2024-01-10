%% Create visualizations for computed L_inf results

% We are interested in:
% 1) How many complete molecules are completely robustly verified (all atoms in a moolecule)?
% 2) How many atoms are robustly verified?

epsilon = [0.005; 0.01; 0.02; 0.05];
eN = length(epsilon);

molecules = zeros(eN,4); % # robust, #unknown, # not robust/misclassified, # molecules
atoms = zeros(eN,4);     % # robust, #unknown, # not robust/misclassified, # atoms

for k = 1:eN
    
    % Load data one at a time
    load("verified_nodes_"+string(epsilon(k))+".mat");

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

save("summary_results_Linf.mat", "atoms", "molecules");