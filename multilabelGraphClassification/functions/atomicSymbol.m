function [symbol,count] = atomicSymbol(atomicNum)
% ATOMICSYMBOL Convert atomic number to symbol
%   symbol = atomicSymbol(atomicNum) returns the atomic symbol of the
%   specified atomic number.
%
%   [symbol,count] = atomicSymbol(atomicNum) also returns the count for
%   each symbol.
%
%   The function supports symbols "H", "C", "N", "O", and "S" only.

numSymbols = numel(atomicNum);
symbol = strings(numSymbols, 1);
count = strings(numSymbols,1);

hCount = 0;
cCount = 0;
nCount = 0;
oCount = 0;
sCount = 0;

for i = 1:numSymbols
    switch atomicNum(i)
        case 1
            symbol(i) = "H";
            hCount = hCount + 1;
            count(i) = "H" + hCount;
        case 6
            symbol(i) = "C";
            cCount = cCount + 1;
            count(i) = "C" + cCount;
        case 7
            symbol(i) = "N";
            nCount = nCount + 1;
            count(i) = "N" + nCount;
        case 8
            symbol(i) = "O";
            oCount = oCount+1;
            count(i) = "O" + oCount;
        otherwise
            symbol(i) = "S";
            sCount = sCount + 1;
            count(i) = "S" + sCount;
    end
end

end
