
#include <mex.h>
#include "matrix.h"

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

#include "preprocessor_definitions.h"

#include "ThrustSystem.h"
#include "Grids.h"
#include "Functors.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // prhs[0] ... command string
  // prhs[1] ... grid type as string
  // prhs[2] ... dimension
  // prhs[3] ... input arguments for constructors
  // prhs[4] ... other input arguments
  
  // Get the command string
  char command[64];
  if (nrhs < 1 || mxGetString(prhs[0], command, sizeof(command) + 1)) {
    mexErrMsgTxt("First input should be a command string less than 64 characters long.");
  }
  if (nrhs < 2) {
    mexErrMsgTxt("Second input should be a string containing the grid type.");
  }
  if (nrhs < 3) {
    mexErrMsgTxt("Third input should be the dimension.");
  }
  if (nrhs < 4) {
    mexErrMsgTxt("Fourth input should be a vector containing the input arguments for the constructors.");
  }
  
  char gridType[255];
  mxGetString(prhs[1], gridType, sizeof(gridType) + 1);
  
  uint64_t DIM = *((uint64_t *) mxGetData(prhs[2]));
  
  char REAL[255];
  
  uint64_t * gridParams = (uint64_t *) mxGetData(prhs[3]);
  int nGridParams = mxGetNumberOfElements(prhs[3]);
  
  // points
  if (!strcmp("points", command)) {
    if (*((bool *) mxGetData(prhs[5]))) {
      // Allocate for output argument
      plhs[0] = mxCreateDoubleMatrix(DIM, mxGetM(prhs[4]), mxREAL);
      strcpy(REAL, "double");
    } else {
      // Allocate for output argument
      plhs[0] = mxCreateNumericMatrix(DIM, mxGetM(prhs[4]), mxSINGLE_CLASS, mxREAL);
      strcpy(REAL, "float");
    }
    char parallelMethod[255];
    mxGetString(prhs[6], parallelMethod, sizeof(parallelMethod) + 1);
    if (0) {
#if USE_CPP != -1
    } else if (!strcmp("cpu", parallelMethod)) {
      if (0) {
      }
#define LOCAL_DIMS_REALS_GRIDS_MACRO(dim, real, grid, n_grid_params)                                    \
  thrust::transform(typename b12::ThrustSystem<b12::CPP>::execution_policy(),                           \
                    (uint64_t *) mxGetData(prhs[4]), (uint64_t *) mxGetData(prhs[4]) + mxGetM(prhs[4]), \
                    thrust::make_transform_iterator(                                                    \
                        thrust::make_transform_iterator(                                                \
                            thrust::make_counting_iterator(b12::NrPoints(0)),                           \
                            thrust::placeholders::_1 * b12::NrPoints(DIM)),                             \
                        b12::AdvanceRealPointerFunctor<real>(                                           \
                            (real *) mxGetData(plhs[0]))),                                              \
                    thrust::make_discard_iterator(),                                                    \
                    b12::grid<dim, real>(N_PARAMS(n_grid_params, gridParams)));
ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_GRIDS
#undef LOCAL_DIMS_REALS_GRIDS_MACRO
      else {
        mexErrMsgTxt("Invalid grid scheme, e.g. wrong grid name or number of parameters, or dimension.");
      }
#endif
#if USE_OMP != -1
    } else if (!strcmp("omp", parallelMethod)) {
      if (0) {
      }
#define LOCAL_DIMS_REALS_GRIDS_MACRO(dim, real, grid, n_grid_params)                                    \
  thrust::transform(typename b12::ThrustSystem<b12::OMP>::execution_policy(),                           \
                    (uint64_t *) mxGetData(prhs[4]), (uint64_t *) mxGetData(prhs[4]) + mxGetM(prhs[4]), \
                    thrust::make_transform_iterator(                                                    \
                        thrust::make_transform_iterator(                                                \
                            thrust::make_counting_iterator(b12::NrPoints(0)),                           \
                            thrust::placeholders::_1 * b12::NrPoints(DIM)),                             \
                        b12::AdvanceRealPointerFunctor<real>(                                           \
                            (real *) mxGetData(plhs[0]))),                                              \
                    thrust::make_discard_iterator(),                                                    \
                    b12::grid<dim, real>(N_PARAMS(n_grid_params, gridParams)));
ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_GRIDS
#undef LOCAL_DIMS_REALS_GRIDS_MACRO
      else {
        mexErrMsgTxt("Invalid grid scheme, e.g. wrong grid name or number of parameters, or dimension.");
      }
#endif
#if USE_TBB != -1
    } else if (!strcmp("tbb", parallelMethod)) {
      if (0) {
      }
#define LOCAL_DIMS_REALS_GRIDS_MACRO(dim, real, grid, n_grid_params)                                    \
  thrust::transform(typename b12::ThrustSystem<b12::TBB>::execution_policy(),                           \
                    (uint64_t *) mxGetData(prhs[4]), (uint64_t *) mxGetData(prhs[4]) + mxGetM(prhs[4]), \
                    thrust::make_transform_iterator(                                                    \
                        thrust::make_transform_iterator(                                                \
                            thrust::make_counting_iterator(b12::NrPoints(0)),                           \
                            thrust::placeholders::_1 * b12::NrPoints(DIM)),                             \
                        b12::AdvanceRealPointerFunctor<real>(                                           \
                            (real *) mxGetData(plhs[0]))),                                              \
                    thrust::make_discard_iterator(),                                                    \
                    b12::grid<dim, real>(N_PARAMS(n_grid_params, gridParams)));
ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_GRIDS
#undef LOCAL_DIMS_REALS_GRIDS_MACRO
      else {
        mexErrMsgTxt("Invalid grid scheme, e.g. wrong grid name or number of parameters, or dimension.");
      }
#endif
#if USE_CUDA != -1
    } else if (!strcmp("gpu", parallelMethod)) {
      if (0) {
      }
#define LOCAL_DIMS_REALS_GRIDS_MACRO(dim, real, grid, n_grid_params)                 \
  typename b12::ThrustSystem<b12::CUDA>::Vector<uint64_t> inds(                      \
    (uint64_t *) mxGetData(prhs[4]),                                                 \
    (uint64_t *) mxGetData(prhs[4]) + mxGetM(prhs[4]));                              \
  typename b12::ThrustSystem<b12::CUDA>::Vector<real> result(DIM * mxGetM(prhs[4])); \
  thrust::transform(typename b12::ThrustSystem<b12::CUDA>::execution_policy(),       \
                    inds.begin(), inds.end(),                                        \
                    thrust::make_transform_iterator(                                 \
                        thrust::make_transform_iterator(                             \
                            thrust::make_counting_iterator(b12::NrPoints(0)),        \
                            thrust::placeholders::_1 * b12::NrPoints(DIM)),          \
                        b12::AdvanceRealPointerFunctor<real>(                        \
                            thrust::raw_pointer_cast(result.data()))),               \
                    thrust::make_discard_iterator(),                                 \
                    b12::grid<dim, real>(N_PARAMS(n_grid_params, gridParams)));      \
  thrust::copy(result.begin(), result.end(), (real *) mxGetData(plhs[0]));
ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_GRIDS
#undef LOCAL_DIMS_REALS_GRIDS_MACRO
      else {
        mexErrMsgTxt("Invalid grid scheme, e.g. wrong grid name or number of parameters, or dimension.");
      }
#endif
    } else {
      mexErrMsgTxt("Invalid parallel method.");
    }
    return;
  }
  
  // nPointsPerBox
  if (!strcmp("nPointsPerBox", command)) {
    // output for output argument
    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    uint64_t * pr = (uint64_t *) mxGetData(plhs[0]);
    strcpy(REAL, "double");
    if (0) {
    }
#define LOCAL_DIMS_REALS_GRIDS_MACRO(dim, real, grid, n_grid_params) \
  pr[0] = b12::grid<dim, real>(N_PARAMS(n_grid_params, gridParams)).getNumberOfPointsPerBox();
ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_GRIDS
#undef LOCAL_DIMS_REALS_GRIDS_MACRO
    else {
      strcpy(REAL, "float");
      if (0) {
      }
#define LOCAL_DIMS_REALS_GRIDS_MACRO(dim, real, grid, n_grid_params) \
  pr[0] = b12::grid<dim, real>(N_PARAMS(n_grid_params, gridParams)).getNumberOfPointsPerBox();
ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_GRIDS
#undef LOCAL_DIMS_REALS_GRIDS_MACRO
      else {
        mexErrMsgTxt("Invalid grid scheme, e.g. wrong grid name or number of parameters, or dimension.");
      }
    }
    return;
  }
  
  // Got here, so command not recognized
  mexErrMsgTxt("Command not recognized.");
}
