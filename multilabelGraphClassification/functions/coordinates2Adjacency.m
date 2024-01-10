function adjacency = coordinates2Adjacency (xyz,atomicNUM)
%coordinates2Adjacency Converts coordinate data to adjacency data

% Input
% xyz   - A 3-by-23-by-N array of atom coordinates, where N is number of
% molecules
% atomicNum   - A 23-by-N array of atomic numbers

% Output
% adjacency   - A 23-by-23-by-N array of adjacency data

% Initialize variables
numMolecules = size(xyz,3);
dist = zeros(size(xyz,1),size(xyz,2),numMolecules);

for molecule = 1:numMolecules
    numAtoms = nonzeros(atomicNUM(:,molecule));
    for atoms = 1:numel(numAtoms)
        dist(atoms,atoms,molecule) = atomicNUM(atoms,molecule);
        j=atoms+1;
        while j <= numel(numAtoms)
            % Calculate the distance as the sum of squres of difference
            % between each coordinate
            dist(atoms,j,molecule) = abs(((xyz(1,atoms,molecule) - (xyz(1,j,molecule))).^2)+((xyz(2,atoms,molecule) ...
                - (xyz(2,j,molecule))).^2)+((xyz(3,atoms,molecule) - (xyz(3,j,molecule))).^2));
            j = j + 1;
        end
    end
end

for molecule = 1:numMolecules
    numAtoms = nonzeros(atomicNUM(:,molecule));
    for atoms = 1:numel(numAtoms)
        j=atoms+1;
        while j <= numel(numAtoms)
        dist(j,atoms,molecule) = dist(atoms,j,molecule);
        j = j+1;
        end
    end
end

% Compute adjacency data
adjacency = zeros(size(xyz,1),size(xyz,2),numMolecules);
leng = zeros(size(xyz,1),size(xyz,2),numMolecules);
dev = zeros(size(xyz,1),size(xyz,2),numMolecules);

% Assess distance between atoms when they are in a bond
for molecule = 1:numMolecules
    numAtoms = nonzeros(atomicNUM(:,molecule));
    atoms = 1;
    while atoms < numel(numAtoms)
        Zi = numAtoms(atoms);
        j = atoms+1;
        while j <= numel(numAtoms)
            Zj = numAtoms(j);
            if atoms == j
                adjacency(atoms,j,molecule) = 0;
            end
            
            index = 1;
            if atoms~=j
                if Zi==16
                    if Zj == 1
                        leng(atoms,j,molecule)=1.34;
                    elseif Zj == 6
                        leng(atoms,j,molecule)=(1.81);
                    elseif Zj == 7
                        leng(atoms,j,molecule)=(1.68);
                    elseif Zj==8
                        leng(atoms,j,molecule)=(1.54);
                    elseif Zj==16
                        leng(atoms,j,molecule)=(2.04);
                    end
                    
                elseif Zi==8
                    if Zj == 1
                        leng(atoms,j,molecule)=0.96;
                    elseif Zj == 6 
                        distances = [1.43 1.23 1.13];
                        [~, index] = min(abs(dist(atoms,j,molecule) - distances));
                        leng(atoms,j,molecule)= distances(index);
                    elseif Zj == 7
                        distances = [1.44 1.20 1.06];
                        [~, index] = min(abs(dist(atoms,j,molecule) - distances));
                        leng(atoms,j,molecule)= distances(index);
                    elseif Zj==8
                        distances = [1.48 1.21];
                        [~, index] = min(abs(dist(atoms,j,molecule) - distances));
                        leng(atoms,j,molecule)= distances(index);
                    elseif Zj==16
                        leng(atoms,j,molecule)=(1.54);
                    end
                    
                elseif Zi == 7
                    if Zj == 1
                        leng(atoms,j,molecule)=1.01;
                    elseif Zj == 6
                        distances = [1.47 1.27 1.15];
                        [~, index] = min(abs(dist(atoms,j,molecule) - distances));
                        leng(atoms,j,molecule)= distances(index);
                    elseif Zj == 7
                       distances = [1.46 1.22 1.1];
                        [~, index] = min(abs(dist(atoms,j,molecule) - distances));
                        leng(atoms,j,molecule)= distances(index);
                    elseif Zj==8
                        distances = [1.44 1.20 1.06];
                        [~, index] = min(abs(dist(atoms,j,molecule) - distances));
                        leng(atoms,j,molecule)= distances(index);
                    elseif Zj==16
                        leng(atoms,j,molecule)=(1.68);
                    end
                
                elseif Zi == 6
                    if Zj == 1
                        leng(atoms,j,molecule)=1.09;
                    elseif Zj == 6
                        distances = [1.54 1.34 1.21];
                        [~, index] = min(abs(dist(atoms,j,molecule) - distances));
                        leng(atoms,j,molecule)= distances(index);
                    elseif Zj == 7
                        distances = [1.47 1.27 1.15];
                        [~, index] = min(abs(dist(atoms,j,molecule) - distances));
                        leng(atoms,j,molecule)= distances(index);
                    elseif Zj==8
                        distances = [1.43 1.23 1.13];
                        [~, index] = min(abs(dist(atoms,j,molecule) - distances));
                        leng(atoms,j,molecule)= distances(index);
                    elseif Zj==16
                        leng(atoms,j,molecule)=(1.81);
                    end
                    
                elseif Zi == 1
                    if Zj == 1
                        leng(atoms,j,molecule)=0.74;
                    elseif Zj == 6
                        leng(atoms,j,molecule)=1.09;
                    elseif Zj == 7
                        leng(atoms,j,molecule)=1.01;
                    elseif Zj==8
                        leng(atoms,j,molecule)=0.96;
                    elseif Zj==16
                        leng(atoms,j,molecule)=1.34;
                    end
                end
                
                % Compare the distance between the atoms with the known
                % length, if the actual distnace is less than the known
                % length then a bond exist                
                dev(atoms,j,molecule) = abs(dist(atoms,j,molecule)-leng(atoms,j,molecule));          
                
                % Compare the given entry in the distance matrix with the
                % known values in length matrix
                if dev(atoms,j,molecule) <= 1
                    adjacency(atoms,j,molecule) = index;
                    adjacency(j,atoms,molecule) = index;
                end
                
            end
            j = j + 1;
        end
        atoms = atoms + 1;
    end
end