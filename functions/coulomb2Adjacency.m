function adjacency = coulomb2Adjacency(loadedColoumb, atomicNumber)

%% Initialize variables
sz = size(loadedColoumb);
adjacency = zeros(sz);
computedColoumb = zeros(sz);

numofMolecules = sz(3);
tolerance = 2.8;

%% Compute coloumb potential
for k = 1:numofMolecules
    Z = nonzeros(atomicNumber(k,:));
    numOfNodes = numel(Z);
    j = 1;
    Zj = Z(j);
    while j < numOfNodes && Zj > 1
        i = j+1;
        while i <= numOfNodes
            if loadedColoumb(i, j, k) > 2.1
                Zi = Z(i);
                if Zj == 16 % Sulphur
                    if Zi == 1 % Hydrogen
                        validIdx = find(loadedColoumb(i:numOfNodes, j, k) > 2.1) + (i - 1);
                        computedColoumb(validIdx,j,k) = iComputeColoumb(Zi, Zj, 1.34);
                        computedColoumb(j,validIdx,k) = computedColoumb(i,j,k);
                        i = numOfNodes + 1;
                    elseif Zi == 6 % Carbon
                        computedColoumb(i,j,k) = iComputeColoumb(Zi, Zj, 1.81);
                        computedColoumb(j,i,k) = computedColoumb(i,j,k);
                        i = i + 1;
                    elseif Zi == 7 % Nitrogen
                        computedColoumb(i,j,k) = iComputeColoumb(Zi, Zj, 1.68);
                        computedColoumb(j,i,k) = computedColoumb(i,j,k);
                        i = i + 1;
                    elseif Zi == 8 % Oxygen
                        computedColoumb(i,j,k) = iComputeColoumb(Zi, Zj, 1.51);
                        computedColoumb(j,i,k) = computedColoumb(i,j,k);
                        i = i + 1;
                    else % Zi == 16
                        computedColoumb(i,j,k) = iComputeColoumb(Zi, Zj, 2.04);
                        computedColoumb(j,i,k) = computedColoumb(i,j,k);
                        i = i + 1;
                    end
                elseif Zj == 8 % Oxygen
                    if Zi == 1 % Hydrogen
                        validIdx = find(loadedColoumb(i:numOfNodes, j, k) > 2.1) + (i - 1);
                        computedColoumb(validIdx,j,k) = iComputeColoumb(Zi, Zj, 0.96);
                        computedColoumb(j,validIdx,k) = computedColoumb(i,j,k);
                        i = numOfNodes + 1;
                    elseif Zi == 6 % Carbon
                        avgLen = (1.43 + 1.23 + 1.13) / 3;
                        computedColoumb(i,j,k) = iComputeColoumb(Zi, Zj, avgLen);
                        computedColoumb(j,i,k) = computedColoumb(i,j,k);
                        i = i + 1;
                    elseif Zi == 7 % Nitrogen
                        avgLen = (1.44 + 1.20 + 1.06) / 3;
                        computedColoumb(i,j,k) = iComputeColoumb(Zi, Zj, avgLen);
                        computedColoumb(j,i,k) = computedColoumb(i,j,k);
                        i = i + 1;
                    else % Zi == 8
                        avgLen = (1.48 + 1.21) / 2;
                        computedColoumb(i,j,k) = iComputeColoumb(Zi, Zj, avgLen);
                        computedColoumb(j,i,k) = computedColoumb(i,j,k);
                        i = i + 1;
                    end
                elseif Zj == 7 % Nitrogen
                    if Zi == 1 % Hydrogen
                        validIdx = find(loadedColoumb(i:numOfNodes, j, k) > 2.1) + (i - 1);
                        computedColoumb(validIdx,j,k) = iComputeColoumb(Zi, Zj, 1.01);
                        computedColoumb(j,validIdx,k) = computedColoumb(i,j,k);
                        i = numOfNodes + 1;
                    elseif Zi == 6 % Carbon
                        avgLen = (1.47 + 1.27 + 1.15)/3;
                        computedColoumb(i,j,k) = iComputeColoumb(Zi, Zj, avgLen);
                        computedColoumb(j,i,k) = computedColoumb(i,j,k);
                        i = i + 1;
                    else % Zi == 7
                        avgLen = (1.46 + 1.22 + 1.1) / 3;
                        computedColoumb(i,j,k) = iComputeColoumb(Zi, Zj, avgLen);
                        computedColoumb(j,i,k) = computedColoumb(i,j,k);
                        i = i + 1;
                    end
                elseif Zj == 6 % Carbon
                    if Zi == 1 % Hydrogen
                        validIdx = find(loadedColoumb(i:numOfNodes, j, k) > 2.1) + (i - 1);
                        computedColoumb(validIdx,j,k) = iComputeColoumb(Zi, Zj, 1.09);
                        computedColoumb(j,validIdx,k) = computedColoumb(i,j,k);
                        i = numOfNodes + 1;
                    else % Zi = 6
                        avgLen = (1.54 + 1.34 + 1.21) / 3;
                        computedColoumb(i,j,k) = iComputeColoumb(Zi, Zj, avgLen);
                        computedColoumb(j,i,k) = computedColoumb(i,j,k);
                        i = i + 1;
                    end
                end
            else
                i = i + 1;
            end
        end
        j = j + 1;
        Zj = Z(j);
    end
end

%% Compute binary adjacency array
absDiff = abs(computedColoumb - loadedColoumb);
isConnected = computedColoumb > 0 & absDiff < tolerance;
adjacency(isConnected) = 1;
end

%% helper
function C = iComputeColoumb(atomicNum1, atomicNum2, bondLength)
atomicLength = 1.88973;
C = (atomicNum1*atomicNum2)/(bondLength*atomicLength);
end
