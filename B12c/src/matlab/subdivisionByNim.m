function et = subdivisionByNim(ibt, gridFunctor, mapFunctor, steps, eigsMethod, filename)

% computing the strange attractor using the subdivision algorithm where the discarding is based on the natural invariant measure.

if nargin < 5
    eigsMethod = 'eigs';
end

scheme = {gridFunctor, mapFunctor, -1};

et = zeros(steps, 2);

HIT = 1;
SD = 8;

% flag all leaves for subdivision
% flag stays even after subdividing and removing
tic;
ibt.setFlags('all', SD);
et(1, 2) = toc + et(1, 2);

for i=1:steps
    
    tic;
    ibt.subdivide; % subdivide flagged leaves
    P = ibt.transitionMatrix(scheme); % compute transition matrix
    et(i, 2) = toc + et(i, 2);
    
    tic;
    if strcmpi(eigsMethod, 'power') || strcmpi(eigsMethod, 'powerMPE')
        l = Inf;
        P = P';
        if gpuDeviceCount > 0
            try
                P = gpuArray(P);
            catch ME
                disp(ME.message);
                warning('Continuing with powerMethodMPE on CPU.');
            end
        end
        while abs(diag(l) - 1) > 0.5 || ~isreal(l)
            [v, l] = powerMethodMPE(P, 1e-14, 1e9);
            v = gather(v);
            l = gather(l);
        end
    else
        ind = true;
        while all(ind)
            [v, l, flag] = eigs(P', min(96, size(P, 2)), 'LM'); % compute eigenvector to largest eigenvalue (~~ 1)
            if flag ~= 0
                disp('Warning: Eigenvector computation did not converge.');
            end
            ind = abs(diag(l) - 1) > 0.5 | imag(diag(l)) ~= 0;
        end
        [l, idx] = max(diag(l .* diag(~ind)));
        v = v(:, idx);
    end
    v = abs(v) ./ norm(v, 1); % normalise eigenvector
    stencil = v >= (1e-9) / length(v); % logical vector indicating which box stays in the tree
    et(i, 1) = toc + et(i, 1);
    
    tic;
    ibt.setFlags(stencil, HIT); % set flag HIT in the leave where stencil[.] is true
    ibt.remove(HIT); % remove boxes which have not been "hit"
    ibt.unsetFlags('all', HIT);
    et(i, 2) = toc + et(i, 2);
    
    if nargin == 6
        tic;
        ibt.save(['output/' filename '_nim_' num2str(ibt.depth, '%02d')]);
        et(i, 2) = toc + et(i, 2);
        
        tic;
        if exist(['output/' filename '_nim_et.txt'], 'file')
            et_temp = load(['output/' filename '_nim_et.txt']);
        else
            et_temp = [];
        end
        et_temp = [et_temp; et(i, :)];
        save(['output/' filename '_nim_et.txt'], 'et_temp', '-ascii');
        et(i, 1) = toc + et(i, 1);
    end
    
    fprintf('depth %2d: %9d boxes, eigenvalue = %12d, Matlab: %9.3d s, B12: %9.3d s\n', ...
            ibt.depth, ibt.count, l, et(i, 1), et(i, 2));
    
end
