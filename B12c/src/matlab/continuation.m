function et = continuation(ibt, gridFunctor, mapFunctor, depths, points, filename)

% continuation algorithm computing instable manifolds starting in points

et = zeros(length(depths), 2);

NONE = 0;
INS = 2;
EXPD = 4;

i = 0;

for d = depths(:)'
    
    i = i + 1;
    
    tic;
    ibt.deleteDepth(0);
    ibt.insert(points, d, INS); % insert initial points
    
    k = 0;
    nb0 = 0;
    tic;
    nb1 = ibt.count(d);
    et(i, 2) = toc + et(i, 2);
    
    scheme = {gridFunctor, mapFunctor, d, EXPD}; % map only points of boxes that have Flags EXPD set
    
    while nb1 > nb0
        tic;
        % previously inserted boxes get EXPD
        ibt.changeFlags('all', INS, EXPD, d);
        % now insert mapped points (new boxes get Flags INS) of previously inserted boxes
        ibt.insert(scheme, d, INS, NONE);
        % once mapped, EXPD is not needed anymore
        ibt.unsetFlags('all', EXPD);
        
        nb0 = nb1;
        nb1 = ibt.count(d);
        et(i, 2) = toc + et(i, 2);

        k = k + 1;
        fprintf('step %5d: %9d boxes\n', k, nb1);
    end
    
    if nargin == 6
        tic;
        ibt.save(['output/' filename '_con_' num2str(d, '%02d')]);
        et(i, 2) = toc + et(i, 2);
        
        tic;
        if exist(['output/' filename '_con_et.txt'], 'file')
            et_temp = load(['output/' filename '_con_et.txt']);
        else
            et_temp = [];
        end
        et_temp = [et_temp; et(i, :)];
        save(['output/' filename '_con_et.txt'], 'et_temp', '-ascii');
        et(i, 1) = toc + et(i, 1);
    end
    
    fprintf('depth %2d: Matlab: %9.3d s, B12: %9.3d s\n', d, et(i, 1), et(i, 2));
end
