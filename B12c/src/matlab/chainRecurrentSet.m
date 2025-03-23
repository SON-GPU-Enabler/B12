function et = chainRecurrentSet(ibt, gridFunctor, mapFunctor, steps, filename)

% computing the chain recurrent set using the subdivision algorithm where boxes are discarded that do not belong to a strongly connected component with at least 2 boxes

scheme = {gridFunctor, mapFunctor, -1};

et = zeros(steps, 2);

HIT = 1;
SD = 8;

% flag all leaves for subdivision
% flag stays even after subdividing and removing
tic;
ibt.setFlags('all', SD);
et(1, 2) = toc + et(1, 2);

for i = 1:steps
    
    tic;
    ibt.subdivide; % subdivide flagged leaves
    P = ibt.transitionMatrix(scheme); % compute transition matrix
    et(i, 2) = toc + et(i, 2);
    
    tic;
    [~, C] = graphconncomp(P'); % compute strongly components in directed graph
    clear P;
    uC = unique(C);
    if length(uC) == 1
        % in this case, histcounts would take uC as the number of bins to use, not a bin center
        counts = length(C);
        centers = uC;
    else
        [counts, centers] = hist(C, uC);
    end
    fprintf(['\nNumber of boxes per SCC:   [', num2str(counts(counts > 1)), ']\n']);
    et(i, 1) = toc + et(i, 1);
    
    tic;
    % set Flags in boxes that belong to a scc that has more than 1 box
    ibt.setFlags(ismember(C, centers(counts > 1)), HIT);
    ibt.remove(HIT); % remove boxes which have not been "hit"
    ibt.unsetFlags('all', HIT);
    et(i, 2) = toc + et(i, 2);
    
    if nargin == 5
        tic;
        ibt.save(['output/' filename '_crs_' num2str(ibt.depth, '%02d')]);
        et(i, 2) = toc + et(i, 2);
        
        tic;
        if exist(['output/' filename '_crs_et.txt'], 'file')
            et_temp = load(['output/' filename '_crs_et.txt']);
        else
            et_temp = [];
        end
        et_temp = [et_temp; et(i, :)];
        save(['output/' filename '_crs_et.txt'], 'et_temp', '-ascii');
        et(i, 1) = toc + et(i, 1);
    end
    
    fprintf('depth %2d: %9d boxes, Matlab: %9.3d s, B12: %9.3d s\n', ibt.depth, ibt.count, et(i, 1), et(i, 2));
    
end
