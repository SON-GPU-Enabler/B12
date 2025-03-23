%% A MATLAB class for a Map functor
% All algorithms are parallelised via OpenMP.
% When input into a gpuImplicitBoxTree, all algorithms are parallelised via CUDA.
% Note that the assignment operator only acts as a copy-by-reference.

classdef MapFunctor < handle
    properties
        mapType; % Map, e.g. Identity, Henon
        dim; % Dimension of the map
        params; % Input arguments for constructors
    end
    methods
        %% Constructor - Create a new C++ class instance 
        function this = MapFunctor(varargin)
            % MapFunctor(mapType, dim, params) creates a dim-dimensional map with params as constructor variables
            if nargin == 1 && isa(varargin{1}, 'MapFunctor')
                this.mapType = varargin{1}.mapType;
                this.dim = varargin{1}.dim;
                this.params = varargin{1}.params;
            else
                this.mapType = varargin{1};
                this.dim = varargin{2};
                if nargin < 3
                    this.params = [];
                else
                    this.params = varargin{3};
                end
            end
        end

        %% Member functions
        
        function res = map(this, points, parallelMethod)
            % returns a (dim,size(points,2))-matrix containing the corresponding mapped points; the computation is done sequentially on CPU if parallelMethod (default='cpu') is 'cpu', in parallel on CPU if it is 'omp', and on GPU if it is 'gpu'
            if nargin < 3
                parallelMethod = 'cpu';
            end
            assert(size(points, 1) == this.dim, 'First argument must have exactly dim rows.');
            assert(ischar(parallelMethod), 'Second argument must be a character array/string.');
            res = map_functor_interface_mex('map', this.mapType, uint64(this.dim), double(this.params), points, parallelMethod);
        end
        
        function res = orbit(this, points, steps, parallelMethod)
            % returns a (dim,size(points,2),steps)-matrix. The first entry along the third dimension equals points, while each other entry are the mapped points of the last entry; the computation is done sequentially on CPU if parallelMethod (default='cpu') is 'cpu', in parallel on CPU if it is 'omp', and on GPU if it is 'cuda'
            if nargin < 4
                parallelMethod = 'cpu';
            end
            assert(size(points, 1) == this.dim, 'First argument must have exactly dim rows.');
            assert(isnumeric(steps) && isscalar(steps) && steps >= 1, 'Second argument must be numeric, scalar, and at least 1.');
            assert(ischar(parallelMethod), 'Third argument must be a character array/string.');
            res = map_functor_interface_mex('orbit', this.mapType, uint64(this.dim), double(this.params), points, double(steps), parallelMethod);
        end
    end
end