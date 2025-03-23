
#include <limits>
#include <random>

#include <curand_kernel.h>

#include "mathFunctions.h"


namespace b12 {

template<Dimension DIM, typename REAL>
__host__ __device__
inline FaceGrid<DIM, REAL>::FaceGrid(NrPoints nPointsPerDim, REAL eps) : nPointsPerDim_(nPointsPerDim), eps_(eps) {}

template<Dimension DIM, typename REAL>
__host__
inline FaceGrid<DIM, REAL>::FaceGrid(NrPoints nPointsPerDim, NrPoints factorEps)
    : nPointsPerDim_(nPointsPerDim), eps_(REAL(factorEps) * std::numeric_limits<REAL>::epsilon()) {}

template<Dimension DIM, typename REAL>
__host__ __device__
inline bool FaceGrid<DIM, REAL>::operator()(NrPoints i, REAL * res) const
{
  NrPoints nPointsPerDimTemp(1), nPointsPerFace(ipow<DIM - 1>(nPointsPerDim_));
#pragma unroll
  for (Dimension j = 0; j < DIM; ++j) {
    if (j == DIM - Dimension((i / (NrPoints(2) * nPointsPerFace)) % NrPoints(DIM)) - Dimension(1)) {
      // -1 or 1 indicating the position of the (DIM - 1)-dimensional face
      res[j] = REAL((i / nPointsPerFace) % NrPoints(2)) * REAL(2.0) - REAL(1.0);
    } else {
      // FullGrid computation
      res[j] = ((i / nPointsPerDimTemp) % nPointsPerDim_) * REAL(2.0) / (nPointsPerDim_ - 1) - REAL(1.0);
      nPointsPerDimTemp *= nPointsPerDim_;
    }
    res[j] *= REAL(1.0) + eps_; // stretch interval from [-1, 1] to [-1 - eps, 1 + eps]
  }
  
  return false;
}

template<Dimension DIM, typename REAL>
__host__ __device__
inline Dimension FaceGrid<DIM, REAL>::dim() const
{
  return DIM;
}

template<Dimension DIM, typename REAL>
__host__ __device__
inline NrPoints FaceGrid<DIM, REAL>::getNumberOfPointsPerBox() const
{
  return ipow<DIM - 1>(nPointsPerDim_) * NrPoints(2 * DIM);
}

template<Dimension DIM, typename REAL>
__host__ __device__
inline FullGrid<DIM, REAL>::FullGrid(NrPoints nPointsPerDim, REAL eps) : nPointsPerDim_(nPointsPerDim), eps_(eps) {}

template<Dimension DIM, typename REAL>
__host__
inline FullGrid<DIM, REAL>::FullGrid(NrPoints nPointsPerDim, NrPoints factorEps)
    : nPointsPerDim_(nPointsPerDim), eps_(REAL(factorEps) * std::numeric_limits<REAL>::epsilon()) {}

template<Dimension DIM, typename REAL>
__host__ __device__
inline bool FullGrid<DIM, REAL>::operator()(NrPoints i, REAL * res) const
{
//   for (Dimension j=0; j<DIM; ++j) {
//     res[j] = ((i / ipow(nPointsPerDim,j)) % nPointsPerDim) * 2.0/(nPointsPerDim-1) - 1.0;
//   }
  
  // probably faster than upper for-loop due to no power computation
  NrPoints nPointsPerDimTemp(1);
#pragma unroll
  for (Dimension j = 0; j < DIM; ++j) {
    res[j] = ((i / nPointsPerDimTemp) % nPointsPerDim_) * REAL(2.0) / (nPointsPerDim_ - 1) - REAL(1.0);
    res[j] *= REAL(1.0) + eps_; // stretch interval from [-1, 1] to [-1 - eps, 1 + eps]
    nPointsPerDimTemp *= nPointsPerDim_;
  }
  
  return false;
}

template<Dimension DIM, typename REAL>
__host__ __device__
inline Dimension FullGrid<DIM, REAL>::dim() const
{
  return DIM;
}

template<Dimension DIM, typename REAL>
__host__ __device__
inline NrPoints FullGrid<DIM, REAL>::getNumberOfPointsPerBox() const
{
  return ipow<DIM>(nPointsPerDim_);
}

template<Dimension DIM, typename REAL>
__host__ __device__
inline InnerGrid<DIM, REAL>::InnerGrid(NrPoints nPointsPerDim) : nPointsPerDim_(nPointsPerDim) {}

template<Dimension DIM, typename REAL>
__host__ __device__
inline bool InnerGrid<DIM, REAL>::operator()(NrPoints i, REAL * res) const
{
//   for (Dimension j=0; j<DIM; ++j) {
//     res[j] = (((i / ipow(nPointsPerDim,j)) % nPointsPerDim)*2.0 + 1.0 - nPointsPerDim) / nPointsPerDim;
//   }
  
  // probably faster than upper for-loop due to no power computation
  NrPoints nPointsPerDimTemp(1);
#pragma unroll
  for (Dimension j = 0; j < DIM; ++j) {
    res[j] = (((i / nPointsPerDimTemp) % nPointsPerDim_) * REAL(2.0) + REAL(1.0) - nPointsPerDim_) / nPointsPerDim_;
    nPointsPerDimTemp *= nPointsPerDim_;
  }
  
  return false;
}

template<Dimension DIM, typename REAL>
__host__ __device__
inline Dimension InnerGrid<DIM, REAL>::dim() const
{
  return DIM;
}

template<Dimension DIM, typename REAL>
__host__ __device__
inline NrPoints InnerGrid<DIM, REAL>::getNumberOfPointsPerBox() const
{
  return ipow<DIM>(nPointsPerDim_);
}

template<Dimension DIM, typename REAL>
__host__ __device__
inline UniformGrid<DIM, REAL>::UniformGrid(NrPoints nPoints, unsigned long long seed)
    : nPoints_(nPoints), seed_(seed) {}

template<Dimension DIM, typename REAL>
__host__ __device__
inline bool UniformGrid<DIM, REAL>::operator()(NrPoints i, REAL * res) const
{
#ifdef __CUDA_ARCH__
  curandStatePhilox4_32_10_t state;
  curand_init(seed_, i, 0, &state);
#pragma unroll
  for (Dimension j = 0; j < DIM; ++j) {
    res[j] = curand_uniform_real<REAL>(&state, REAL(-1.0), REAL(1.0));
  }
#else
  std::mt19937 gen(seed_ + i);
  std::uniform_real_distribution<REAL> dis(REAL(-1.0), REAL(1.0));
#pragma unroll
  for (Dimension j = 0; j < DIM; ++j) {
    res[j] = dis(gen);
  }
#endif
  
  return false;
}

template<Dimension DIM, typename REAL>
__host__ __device__
inline Dimension UniformGrid<DIM, REAL>::dim() const
{
  return DIM;
}

template<Dimension DIM, typename REAL>
__host__ __device__
inline NrPoints UniformGrid<DIM, REAL>::getNumberOfPointsPerBox() const
{
  return nPoints_;
}

} // namespace b12
