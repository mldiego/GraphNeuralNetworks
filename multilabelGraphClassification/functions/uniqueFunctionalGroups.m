function labels = uniqueFunctionalGroups(adjacency, atomicNumber)
%uniqueFunctionalGroups   Get unique functional groups

% Load functional group of interest
load('functionalGroupsOfInterest.mat')

numGraphs = size(adjacency,3);
labels = cell(numGraphs, 1);

for i = 1:numGraphs
    [modifiedAdjacency, modifiedAtomics] = preprocess(adjacency(:,:,i), atomicNumber(:,i));
    foundFGs = ertlFinder(modifiedAtomics, modifiedAdjacency);
    smiles = smilesConverter(foundFGs, functionalGroupsOfInterest);
    labels{i} = unique(smiles);
end
end

%%
function [modifiedAdjacency, modifiedAtomics] = preprocess(adjacencyData, atomicData)
    % Swap atomic numbers for their string equivalent and remove zeros used
    % to pad the data
    nonzeroAtomicData = nonzeros(atomicData);
    modifiedAtomics = [];
    for i = 1:length(nonzeroAtomicData)
        if nonzeroAtomicData(i) == 1
            modifiedAtomics = [modifiedAtomics, "Hydrogen"];
        elseif nonzeroAtomicData(i) == 6
            modifiedAtomics = [modifiedAtomics, "Carbon"];
        elseif nonzeroAtomicData(i) == 7
            modifiedAtomics = [modifiedAtomics, "Nitrogen"];
        elseif nonzeroAtomicData(i) == 8
            modifiedAtomics = [modifiedAtomics, "Oxygen"];
        elseif nonzeroAtomicData(i) == 16
            modifiedAtomics = [modifiedAtomics, "Sulfur"];
        end
    end
    
    numAtoms = length(modifiedAtomics);
    modifiedAdjacency = adjacencyData(1:numAtoms, 1:numAtoms);
end

%%
function functionalGroups = ertlFinder(atoms, adjacency)
% etrl algorithm based on https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0225-z
% Steps:
% 1. Mark all heteroatoms
% 2. Mark carbon atoms
%   a. Atoms connected by non-aromatic double or triple bond to any heteroatom
%   b. Atoms in nonaromatic carbon–carbon double or triple bonds
%   c. Acetal carbons, i.e. sp3 carbons connected to two or more oxygens,
%   nitrogens or sulfurs. These O, N or S atoms must have only single bonds 
% 3. Merge all connected marked atoms to a single functional group
markedAtoms = [];
functionalGroups = [];

% Step 1 - Mark the heteroatoms
for i = 1:size(adjacency, 1)
    if (atoms(i) ~= "Carbon") && (atoms(i) ~= "Hydrogen")
        % all non-hydrogen and carbon atoms are heteroatoms
        if ismember(i, markedAtoms) == false
            markedAtoms = [markedAtoms, i];
        end
    end
end

% Extra step - original algorithm doesn't track hydrogens, we will in our
% case
for i = markedAtoms
    for j = 1:size(adjacency, 1) % loop through all atoms
        if i ~= j && adjacency(i,j) >= 1 && (atoms(j) == "Hydrogen")
            if ismember(j, markedAtoms) == 0
                    markedAtoms = [markedAtoms, j];
            end
        end
    end
end

% Step 2a - Carbon atoms connected to heteroatoms
for i = markedAtoms
    for j = 1:size(adjacency, 1)
        if i ~= j && adjacency(i,j) > 1
            if ismember(j, markedAtoms) == 0
                    markedAtoms = [markedAtoms, j];
            end
        end
    end
end

% Step 2b - Mark all carbon atoms that are in double/triple bonds with
% other carbon atoms
for i = 1:size(adjacency, 1)
    for j = 1:size(adjacency, 1)
         if i ~= j && atoms(i) == "Carbon" && atoms(j) == "Carbon" ...
                 && adjacency(i,j) > 1
            if ismember(j, markedAtoms) == false 
                    markedAtoms = [markedAtoms, j];
            end
         end
    end
end

% Step 2c -Mark acetals
for i = 1:size(adjacency, 1)
    count = 0;
    potentialAtoms = i;
    if atoms(i) == "Carbon"
        for j = 1:size(adjacency, 1)
            if i ~= j && adjacency(i,j) == 1 && atoms(j) ~= "Carbon" && ...
                    atoms(j) ~= "Hydrogen"
                count = count + 1;
                potentialAtoms = [potentialAtoms, j];
            end   
        end
    end
    
    if count > 1
        for j = 1:length(potentialAtoms)
            if ismember(j, markedAtoms) == false 
                markedAtoms = [markedAtoms, j];
            end
        end
    end
end

% Extra step: Not in the original ertl algorithm, but we want to add 
% CH3, CH2, and CH as functional groups of interest
for i = 1:size(adjacency, 1)
    if atoms(i) == "Carbon" && ismember(i, markedAtoms) == 0
        hydrogenCount = 0;
        for j = 1:size(adjacency, 1)
            if atoms(j) == "Hydrogen" && adjacency(i,j) > 0 && i ~= j
                hydrogenCount = hydrogenCount + 1;
            end
        end
        
        if hydrogenCount == 1
            functionalGroups = [functionalGroups; "H1C1N0O0S0"];
        elseif hydrogenCount == 2
            functionalGroups = [functionalGroups; "H2C1N0O0S0"];
        elseif hydrogenCount == 3
            functionalGroups = [functionalGroups; "H3C1N0O0S0"];
        end
        
    end
end
% Step 3 - Merge connected atoms
functionalGroupsIndices = [];
while isempty(markedAtoms) == 0
    stop = 0;
    currentFG = [markedAtoms(1)];
    while stop == 0
        stop = 1;
        for j = markedAtoms
            for k = currentFG        
                if j ~= k && adjacency(j,k) > 0 && ismember(j, currentFG) == 0
                    currentFG = [currentFG, j];
                    stop = 0;
                end
            end
        end
    end
    
    for j = currentFG
        markedAtoms(markedAtoms==j) = [];
    end
    functionalGroupsIndices = [functionalGroupsIndices, {currentFG}];
end

for i = 1:length(functionalGroupsIndices)
    fg = functionalGroupsIndices{i};
    
    carbons = 0;
    hydrogens = 0;
    nitrogens = 0;
    sulfurs = 0;
    oxygens = 0;
    
    for j = flip(fg)
        if atoms(j) == "Carbon"
            carbons = carbons + 1;
        elseif atoms(j) == "Hydrogen"
            hydrogens = hydrogens + 1;
        elseif atoms(j) == "Sulfur"
            sulfurs = sulfurs + 1;
        elseif atoms(j) == "Oxygen"
            oxygens = oxygens + 1;
        elseif atoms(j) == "Nitrogen"
            nitrogens = nitrogens + 1;
        end
    end
    
    functionalGroups = [functionalGroups; ...
        strcat("H", num2str(hydrogens), ...
        "C", num2str(carbons), ...
        "N", num2str(nitrogens), ...
        "O", num2str(oxygens), ...
        "S", num2str(sulfurs))];
end
end
%%
function smiles = smilesConverter(functionalGroups, lookUp)
smiles = [];
for i = 1:length(functionalGroups)
    idx = strfind(lookUp.elementCount, functionalGroups(i));
    idx = find(not(cellfun('isempty',idx)));
    smiles = [smiles; lookUp.smiles(idx)];
end
end