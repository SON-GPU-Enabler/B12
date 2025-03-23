
#include <mex.h>
#include "matrix.h"

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

#include "preprocessor_definitions.h"

#include "ThrustSystem.h"
#include "Maps.h"
#include "Functors.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // prhs[0] ... command string
  // prhs[1] ... map name as string
  // prhs[2] ... dimension
  // prhs[3] ... input arguments for constructors
  // prhs[4] ... points to map
  
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
  if (nrhs < 5) {
    mexErrMsgTxt("Fifth input should be a matrix containing the points to map.");
  }
  
  char mapType[255];
  mxGetString(prhs[1], mapType, sizeof(mapType) + 1);
  
  uint64_t DIM = *((uint64_t *) mxGetData(prhs[2]));
  
  char REAL[255];
  
  double * mapParams = mxGetPr(prhs[3]);
  int nMapParams = mxGetNumberOfElements(prhs[3]);
  
  // map points
  if (!strcmp("map", command)) {
    if (mxIsDouble(prhs[4])) {
      // Allocate for output argument
      plhs[0] = mxCreateDoubleMatrix(DIM, mxGetN(prhs[4]), mxREAL);
      strcpy(REAL, "double");
    } else {
      plhs[0] = mxCreateNumericMatrix(DIM, mxGetN(prhs[4]), mxSINGLE_CLASS, mxREAL);
      strcpy(REAL, "float");
    }
    char parallelMethod[255];
    mxGetString(prhs[5], parallelMethod, sizeof(parallelMethod) + 1);
    if (0) {
#if USE_CPP != -1
    } else if (!strcmp("cpu", parallelMethod)) {
      if (0) {
      }
#define LOCAL_DIMS_REALS_MAPS_MACRO(dim, real, map, n_map_params)             \
  thrust::transform(typename b12::ThrustSystem<b12::CPP>::execution_policy(), \
                    thrust::make_transform_iterator(                          \
                        thrust::make_transform_iterator(                      \
                            thrust::make_counting_iterator(b12::NrPoints(0)), \
                            thrust::placeholders::_1 * b12::NrPoints(DIM)),   \
                        b12::AdvanceConstRealPointerFunctor<real>(            \
                              (real *) mxGetData(prhs[4]))),                  \
                    thrust::make_transform_iterator(                          \
                        thrust::make_transform_iterator(                      \
                            thrust::make_counting_iterator(b12::NrPoints(0)), \
                            thrust::placeholders::_1 * b12::NrPoints(DIM)),   \
                        b12::AdvanceConstRealPointerFunctor<real>(            \
                            (real *) mxGetData(prhs[4]))) + mxGetN(prhs[4]),  \
                    thrust::make_transform_iterator(                          \
                        thrust::make_transform_iterator(                      \
                            thrust::make_counting_iterator(b12::NrPoints(0)), \
                            thrust::placeholders::_1 * b12::NrPoints(DIM)),   \
                        b12::AdvanceRealPointerFunctor<real>(                 \
                            (real *) mxGetData(plhs[0]))),                    \
                    thrust::make_discard_iterator(),                          \
                    b12::map<dim, real>(N_PARAMS(n_map_params, mapParams)));
ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAPS
#undef LOCAL_DIMS_REALS_MAPS_MACRO
      else {
        mexErrMsgTxt("Invalid map scheme, e.g. wrong map name or number of parameters, or dimension.");
      }
#endif
#if USE_OMP != -1
    } else if (!strcmp("omp", parallelMethod)) {
      if (0) {
      }
#define LOCAL_DIMS_REALS_MAPS_MACRO(dim, real, map, n_map_params)             \
  thrust::transform(typename b12::ThrustSystem<b12::OMP>::execution_policy(), \
                    thrust::make_transform_iterator(                          \
                        thrust::make_transform_iterator(                      \
                            thrust::make_counting_iterator(b12::NrPoints(0)), \
                            thrust::placeholders::_1 * b12::NrPoints(DIM)),   \
                        b12::AdvanceConstRealPointerFunctor<real>(            \
                              (real *) mxGetData(prhs[4]))),                  \
                    thrust::make_transform_iterator(                          \
                        thrust::make_transform_iterator(                      \
                            thrust::make_counting_iterator(b12::NrPoints(0)), \
                            thrust::placeholders::_1 * b12::NrPoints(DIM)),   \
                        b12::AdvanceConstRealPointerFunctor<real>(            \
                            (real *) mxGetData(prhs[4]))) + mxGetN(prhs[4]),  \
                    thrust::make_transform_iterator(                          \
                        thrust::make_transform_iterator(                      \
                            thrust::make_counting_iterator(b12::NrPoints(0)), \
                            thrust::placeholders::_1 * b12::NrPoints(DIM)),   \
                        b12::AdvanceRealPointerFunctor<real>(                 \
                            (real *) mxGetData(plhs[0]))),                    \
                    thrust::make_discard_iterator(),                          \
                    b12::map<dim, real>(N_PARAMS(n_map_params, mapParams)));
ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAPS
#undef LOCAL_DIMS_REALS_MAPS_MACRO
      else {
        mexErrMsgTxt("Invalid map scheme, e.g. wrong map name or number of parameters, or dimension.");
      }
#endif
#if USE_TBB != -1
    } else if (!strcmp("tbb", parallelMethod)) {
      if (0) {
      }
#define LOCAL_DIMS_REALS_MAPS_MACRO(dim, real, map, n_map_params)             \
  thrust::transform(typename b12::ThrustSystem<b12::TBB>::execution_policy(), \
                    thrust::make_transform_iterator(                          \
                        thrust::make_transform_iterator(                      \
                            thrust::make_counting_iterator(b12::NrPoints(0)), \
                            thrust::placeholders::_1 * b12::NrPoints(DIM)),   \
                        b12::AdvanceConstRealPointerFunctor<real>(            \
                              (real *) mxGetData(prhs[4]))),                  \
                    thrust::make_transform_iterator(                          \
                        thrust::make_transform_iterator(                      \
                            thrust::make_counting_iterator(b12::NrPoints(0)), \
                            thrust::placeholders::_1 * b12::NrPoints(DIM)),   \
                        b12::AdvanceConstRealPointerFunctor<real>(            \
                            (real *) mxGetData(prhs[4]))) + mxGetN(prhs[4]),  \
                    thrust::make_transform_iterator(                          \
                        thrust::make_transform_iterator(                      \
                            thrust::make_counting_iterator(b12::NrPoints(0)), \
                            thrust::placeholders::_1 * b12::NrPoints(DIM)),   \
                        b12::AdvanceRealPointerFunctor<real>(                 \
                            (real *) mxGetData(plhs[0]))),                    \
                    thrust::make_discard_iterator(),                          \
                    b12::map<dim, real>(N_PARAMS(n_map_params, mapParams)));
ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAPS
#undef LOCAL_DIMS_REALS_MAPS_MACRO
      else {
        mexErrMsgTxt("Invalid map scheme, e.g. wrong map name or number of parameters, or dimension.");
      }
#endif
#if USE_CUDA != -1
    } else if (!strcmp("gpu", parallelMethod)) {
      if (0) {
      }
#define LOCAL_DIMS_REALS_MAPS_MACRO(dim, real, map, n_map_params)                        \
  typename b12::ThrustSystem<b12::CUDA>::Vector<real> points(                            \
    (real *) mxGetData(prhs[4]),                                                         \
    (real *) mxGetData(prhs[4]) + DIM * mxGetN(prhs[4]));                                \
  typename b12::ThrustSystem<b12::CUDA>::Vector<real> result(points.size());             \
  thrust::transform(typename b12::ThrustSystem<b12::CUDA>::execution_policy(),           \
                    thrust::make_transform_iterator(                                     \
                        thrust::make_transform_iterator(                                 \
                            thrust::make_counting_iterator(b12::NrPoints(0)),            \
                            thrust::placeholders::_1 * b12::NrPoints(DIM)),              \
                        b12::AdvanceConstRealPointerFunctor<real>(                       \
                              thrust::raw_pointer_cast(points.data()))),                 \
                    thrust::make_transform_iterator(                                     \
                        thrust::make_transform_iterator(                                 \
                            thrust::make_counting_iterator(b12::NrPoints(0)),            \
                            thrust::placeholders::_1 * b12::NrPoints(DIM)),              \
                        b12::AdvanceConstRealPointerFunctor<real>(                       \
                            thrust::raw_pointer_cast(points.data()))) + mxGetN(prhs[4]), \
                    thrust::make_transform_iterator(                                     \
                        thrust::make_transform_iterator(                                 \
                            thrust::make_counting_iterator(b12::NrPoints(0)),            \
                            thrust::placeholders::_1 * b12::NrPoints(DIM)),              \
                        b12::AdvanceRealPointerFunctor<real>(                            \
                            thrust::raw_pointer_cast(result.data()))),                   \
                    thrust::make_discard_iterator(),                                     \
                    b12::map<dim, real>(N_PARAMS(n_map_params, mapParams)));             \
  thrust::copy(result.begin(), result.end(), (real *) mxGetData(plhs[0]));
ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAPS
#undef LOCAL_DIMS_REALS_MAPS_MACRO
      else {
        mexErrMsgTxt("Invalid map scheme, e.g. wrong map name or number of parameters, or dimension.");
      }
#endif
    } else {
      mexErrMsgTxt("Invalid parallel method.");
    }
    return;
  }
  
  // create orbit of points
  if (!strcmp("orbit", command)) {
    mwSize dimensions[3];
    dimensions[0] = mxGetM(prhs[4]);
    dimensions[1] = mxGetN(prhs[4]);
    dimensions[2] = * mxGetPr(prhs[5]);
    if (mxIsDouble(prhs[4])) {
      // Allocate for output argument
      plhs[0] = mxCreateNumericArray(3, dimensions, mxDOUBLE_CLASS, mxREAL);
      strcpy(REAL, "double");
    } else {
      plhs[0] = mxCreateNumericArray(3, dimensions, mxSINGLE_CLASS, mxREAL);
      strcpy(REAL, "float");
    }
    char parallelMethod[255];
    mxGetString(prhs[6], parallelMethod, sizeof(parallelMethod) + 1);
    if (0) {
#if USE_CPP != -1
    } else if (!strcmp("cpu", parallelMethod)) {
      if (0) {
      }
#define LOCAL_DIMS_REALS_MAPS_MACRO(dim, real, map, n_map_params)               \
  thrust::copy((real *) mxGetData(prhs[4]),                                     \
               (real *) mxGetData(prhs[4]) + dimensions[0] * dimensions[1],     \
               (real *) mxGetData(plhs[0]));                                    \
  auto it = thrust::make_transform_iterator(                                    \
      thrust::make_transform_iterator(                                          \
          thrust::make_counting_iterator(b12::NrPoints(0)),                     \
          thrust::placeholders::_1 * b12::NrPoints(DIM)),                       \
      b12::AdvanceRealPointerFunctor<real>((real *) mxGetData(plhs[0])));       \
  for (int i = 1; i < dimensions[2]; ++i, it += dimensions[1]) {                \
    thrust::transform(typename b12::ThrustSystem<b12::CPP>::execution_policy(), \
                      it, it + dimensions[1],                                   \
                      it + dimensions[1],                                       \
                      thrust::make_discard_iterator(),                          \
                      b12::map<dim, real>(N_PARAMS(n_map_params, mapParams)));  \
  }
ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAPS
#undef LOCAL_DIMS_REALS_MAPS_MACRO
      else {
        mexErrMsgTxt("Invalid map scheme, e.g. wrong map name or number of parameters, or dimension.");
      }
#endif
#if USE_OMP != -1
    } else if (!strcmp("omp", parallelMethod)) {
      if (0) {
      }
#define LOCAL_DIMS_REALS_MAPS_MACRO(dim, real, map, n_map_params)               \
  thrust::copy((real *) mxGetData(prhs[4]),                                     \
               (real *) mxGetData(prhs[4]) + dimensions[0] * dimensions[1],     \
               (real *) mxGetData(plhs[0]));                                    \
  auto it = thrust::make_transform_iterator(                                    \
      thrust::make_transform_iterator(                                          \
          thrust::make_counting_iterator(b12::NrPoints(0)),                     \
          thrust::placeholders::_1 * b12::NrPoints(DIM)),                       \
      b12::AdvanceRealPointerFunctor<real>((real *) mxGetData(plhs[0])));       \
  for (int i = 1; i < dimensions[2]; ++i, it += dimensions[1]) {                \
    thrust::transform(typename b12::ThrustSystem<b12::OMP>::execution_policy(), \
                      it, it + dimensions[1],                                   \
                      it + dimensions[1],                                       \
                      thrust::make_discard_iterator(),                          \
                      b12::map<dim, real>(N_PARAMS(n_map_params, mapParams)));  \
  }
ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAPS
#undef LOCAL_DIMS_REALS_MAPS_MACRO
      else {
        mexErrMsgTxt("Invalid map scheme, e.g. wrong map name or number of parameters, or dimension.");
      }
#endif
#if USE_TBB != -1
    } else if (!strcmp("tbb", parallelMethod)) {
      if (0) {
      }
#define LOCAL_DIMS_REALS_MAPS_MACRO(dim, real, map, n_map_params)               \
  thrust::copy((real *) mxGetData(prhs[4]),                                     \
               (real *) mxGetData(prhs[4]) + dimensions[0] * dimensions[1],     \
               (real *) mxGetData(plhs[0]));                                    \
  auto it = thrust::make_transform_iterator(                                    \
      thrust::make_transform_iterator(                                          \
          thrust::make_counting_iterator(b12::NrPoints(0)),                     \
          thrust::placeholders::_1 * b12::NrPoints(DIM)),                       \
      b12::AdvanceRealPointerFunctor<real>((real *) mxGetData(plhs[0])));       \
  for (int i = 1; i < dimensions[2]; ++i, it += dimensions[1]) {                \
    thrust::transform(typename b12::ThrustSystem<b12::TBB>::execution_policy(), \
                      it, it + dimensions[1],                                   \
                      it + dimensions[1],                                       \
                      thrust::make_discard_iterator(),                          \
                      b12::map<dim, real>(N_PARAMS(n_map_params, mapParams)));  \
  }
ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAPS
#undef LOCAL_DIMS_REALS_MAPS_MACRO
      else {
        mexErrMsgTxt("Invalid map scheme, e.g. wrong map name or number of parameters, or dimension.");
      }
#endif
#if USE_CUDA != -1
    } else if (!strcmp("gpu", parallelMethod)) {
      if (0) {
      }
#define LOCAL_DIMS_REALS_MAPS_MACRO(dim, real, map, n_map_params)                                           \
  typename b12::ThrustSystem<b12::CUDA>::Vector<real> orbit(dimensions[0] * dimensions[1] * dimensions[2]); \
  thrust::copy((real *) mxGetData(prhs[4]),                                                                 \
               (real *) mxGetData(prhs[4]) + dimensions[0] * dimensions[1],                                 \
               orbit.begin());                                                                              \
  auto it = thrust::make_transform_iterator(                                                                \
      thrust::make_transform_iterator(                                                                      \
          thrust::make_counting_iterator(b12::NrPoints(0)),                                                 \
          thrust::placeholders::_1 * b12::NrPoints(DIM)),                                                   \
      b12::AdvanceRealPointerFunctor<real>(thrust::raw_pointer_cast(orbit.data())));                        \
  for (int i = 1; i < dimensions[2]; ++i, it += dimensions[1]) {                                            \
    thrust::transform(typename b12::ThrustSystem<b12::CUDA>::execution_policy(),                            \
                      it, it + dimensions[1],                                                               \
                      it + dimensions[1],                                                                   \
                      thrust::make_discard_iterator(),                                                      \
                      b12::map<dim, real>(N_PARAMS(n_map_params, mapParams)));                              \
  }                                                                                                         \
  thrust::copy(orbit.begin(), orbit.end(), (real *) mxGetData(plhs[0]));
ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAPS
#undef LOCAL_DIMS_REALS_MAPS_MACRO
      else {
        mexErrMsgTxt("Invalid map scheme, e.g. wrong map name or number of parameters, or dimension.");
      }
#endif
    } else {
      mexErrMsgTxt("Invalid parallel method.");
    }
    return;
  }
  
  // Got here, so command not recognized
  mexErrMsgTxt("Command not recognized.");
}
