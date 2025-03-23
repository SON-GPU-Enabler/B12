function fbs = fixedBoxes(ibt, gridFunctor, mapFunctor)

% iterative subdividing and discarding boxes that do not hit itself

scheme = {gridFunctor, mapFunctor, -1};

HIT = 1;
SD = 8;

ibt.deleteDepth(0);
ibt.setFlags('all', SD);

for i = 1:ibt.maxDepth
    ibt.subdivide(SD);
    s = ibt.search(scheme);
    s = reshape(s, gridFunctor.nPointsPerBox, []);
    ibt.setFlags(any(s == 1:size(s, 2)), HIT);
    ibt.remove(HIT);
    ibt.unsetFlags('all', HIT);
end

fbs = ibt.boxes;
fbs = fbs(1:end-1, :);
