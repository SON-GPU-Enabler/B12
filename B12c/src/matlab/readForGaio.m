function t = readForGaio(fileName)

fileID = fopen(fileName);

dim = fread(fileID, [1 1], 'int');

size_of_coordinatetype = fread(fileID, [1 1], 'int');

if size_of_coordinatetype == 4
    center = fread(fileID, [dim 1], 'float');
    radius = fread(fileID, [dim 1], 'float');
elseif size_of_coordinatetype == 8
    center = fread(fileID, [dim 1], 'double');
    radius = fread(fileID, [dim 1], 'double');
else
    error('Cannot read RealType');
end
sd = fread(fileID, [dim * 500 + 1, 1], 'int');

t = Tree(center, radius);
t.sd = sd;

nrb = fread(fileID, [1 1], 'double');

for i = 1:nrb
    if size_of_coordinatetype == 4
        v = fread(fileID, [dim 1], 'float');
    elseif size_of_coordinatetype == 8
        v = fread(fileID, [dim 1], 'double');
    else
        error('Cannot read RealType');
    end
    depth = fread(fileID, [1 1], 'double');
    flags = fread(fileID, [1 1], 'double');
    t.insert(v, depth, flags);
end

fclose(fileID);