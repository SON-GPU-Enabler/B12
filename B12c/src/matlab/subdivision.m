function et = subdivision(ibt, gridFunctor, mapFunctor, steps, filename)

% subdivision algorithm computing of the relative global attractor

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
    ibt.subdivide(SD); % subdivide flagged leaves
    ibt.setFlags(scheme, HIT, -1); % flag hitted boxes
    ibt.remove(HIT); % remove boxes which have not been hit
    ibt.unsetFlags('all', HIT);
    et(i, 2) = toc + et(i, 2);
    
    if nargin == 5
        tic;
        ibt.save(['output/' filename '_rga_' num2str(ibt.depth, '%02d')]);
        et(i, 2) = toc + et(i, 2);
        
        tic;
        if exist(['output/' filename '_rga_et.txt'], 'file')
            et_temp = load(['output/' filename '_rga_et.txt']);
        else
            et_temp = [];
        end
        et_temp = [et_temp; et(i, :)];
        save(['output/' filename '_rga_et.txt'], 'et_temp', '-ascii');
        et(i, 1) = toc + et(i, 1);
    end
    
    fprintf('depth %2d: %9d boxes, Matlab: %9.3d s, B12: %9.3d s\n', ibt.depth, ibt.count, et(i, 1), et(i, 2));

end