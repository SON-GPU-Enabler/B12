
#pragma once


// necessary combinations are cmake-compiled into preprocessor_combinations.h
#include "preprocessor_combinations.h"


// expands to: param_field[0], param_field[1], ..., param_field[n_params - 1]
#define N_PARAMS(n_params, param_field) N_PARAMS_ ## n_params (param_field)




// if the input is a combination of a dimension, a real type, and a max. depth
#define ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH(dim, real, maxDepth) \
  else if (DIM == dim &&                                              \
           !strcmp(REAL, #real) &&                                    \
           MAXDEPTH == maxDepth) {                                    \
    LOCAL_DIMS_REALS_MAXDEPTHS_MACRO(dim, real, maxDepth)             \
  }


// if the input is a combination of a dimension, a real type, a max. depth, a dimension, a real type, and a max. depth
#define ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(dim, real, maxDepth, dim2, real2, maxDepth2) \
  else if (DIM == dim &&                                                                                           \
           !strcmp(REAL, #real) &&                                                                                 \
           MAXDEPTH == maxDepth &&                                                                                 \
           DIM2 == dim2 &&                                                                                         \
           !strcmp(REAL2, #real2) &&                                                                               \
           MAXDEPTH2 == maxDepth2) {                                                                               \
    LOCAL_NESTED_DIMS_REALS_MAXDEPTHS_MACRO(dim, real, maxDepth, dim2, real2, maxDepth2)                           \
  }


// if the input is a combination of a dimension, a real type, and a grid
#define ELSE_IF_COMBINATION_OF_DIM_REAL_GRID(dim, real, grid, n_grid_params) \
  else if (DIM == dim &&                                                     \
           !strcmp(REAL, #real) &&                                           \
           !strcmp(gridType, #grid) &&                                       \
           nGridParams == n_grid_params) {                                   \
    LOCAL_DIMS_REALS_GRIDS_MACRO(dim, real, grid, n_grid_params)             \
  }


// if the input is a combination of a dimension, a real type, and a map
#define ELSE_IF_COMBINATION_OF_DIM_REAL_MAP(dim, real, map, n_map_params) \
  else if (DIM == dim &&                                                  \
           !strcmp(REAL, #real) &&                                        \
           !strcmp(mapType, #map) &&                                      \
           nMapParams == n_map_params) {                                  \
    LOCAL_DIMS_REALS_MAPS_MACRO(dim, real, map, n_map_params)             \
  }


// if the input is a combination of a dimension, a real type, a max. depth, a grid, and a map
#define ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_GRID_MAP(dim, real, maxDepth, grid, n_grid_params, map, n_map_params) \
  else if (DIM == dim &&                                                                                               \
           !strcmp(REAL, #real) &&                                                                                     \
           MAXDEPTH == maxDepth &&                                                                                     \
           !strcmp(gridType, #grid) &&                                                                                 \
           nGridParams == n_grid_params &&                                                                             \
           !strcmp(mapType, #map) &&                                                                                   \
           nMapParams == n_map_params) {                                                                               \
    LOCAL_DIMS_REALS_MAXDEPTHS_GRIDS_MAPS_MACRO(dim, real, maxDepth, grid, n_grid_params, map, n_map_params)           \
  }
