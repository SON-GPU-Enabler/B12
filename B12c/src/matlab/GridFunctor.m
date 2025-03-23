%% A MATLAB class for a Grid functor
% All algorithms are parallelised via OpenMP.
% When inputted to a gpuImplicitBoxTree, all algorithms are parallelised via CUDA.
% Note that the assignment operator only acts as a copy-by-reference.

classdef GridFunctor < handle
    properties
        gridType; % gridType for Grid, e.g. FullGrid, InnerGrid
        dim; % Dimension for the Grid
        params; % Input arguments for constructors
    end
    methods
        %% Constructor - Create a new C++ class instance 
        function this = GridFunctor(varargin)
            % GridFunctor(gridType, dim, params) creates a dim-dimensional grid w.r.t. gridType and with params as constructor variables
            if nargin == 1 && isa(varargin{1}, 'GridFunctor')
                this.gridType = varargin{1}.gridType;
                this.dim = varargin{1}.dim;
                this.params = varargin{1}.params;
            else
                this.gridType = varargin{1};
                this.dim = varargin{2};
                if nargin < 3
                    this.params = [];
                else
                    this.params = varargin{3};
                end
            end
        end

        %% Member functions
        
        function res = points(this, vec, useDouble, parallelMethod)
            % returns a (dim, numel(vec))-matrix containing the corresponding grid points with double precision if useDouble (default=true) is true and single precision otherwise; by default: vec == 1:this.nPointsPerBox()
            if nargin < 4
                parallelMethod = 'cpu';
                if nargin < 3
                    useDouble = true;
                    if nargin < 2
                        vec = 1:this.nPointsPerBox();
                    end
                end
            end
            assert(isnumeric(vec) && all(vec >= 1), 'Argument must be numeric and positive.');
            res = grid_functor_interface_mex('points', this.gridType, uint64(this.dim), uint64(this.params), ...
                                             uint64(vec(:)-1), logical(useDouble), parallelMethod);
        end
        
        function res = nPointsPerBox(this)
            % returns the number of points for each box
            res = grid_functor_interface_mex('nPointsPerBox', this.gridType, uint64(this.dim), uint64(this.params));
        end
    end
end