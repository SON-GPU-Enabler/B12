
#pragma once

#include "MapTools.h"


namespace b12 {

template<typename REAL>
struct DuffingMap
{
  REAL rho_, eps_;
  
  __host__ __device__
  inline DuffingMap(REAL rho, REAL eps) : rho_(rho), eps_(eps) {}
  
  __host__ __device__
  inline void operator()(const REAL * point, REAL * res) const
  {
    res[0] = point[0] + eps_ * point[1] + REAL(0.5)*eps_*eps_ * (rho_ * point[0] - point[0]*point[0]*point[0]);
    res[1] = point[1] + 
             REAL(0.5)*eps_ * (rho_ * point[0] - point[0]*point[0]*point[0] + rho_*res[0] - res[0]*res[0]*res[0]);
  }
};


template<Dimension DIM, typename REAL>
struct Duffing : public thrust::binary_function<REAL*, REAL*, bool>
{
  IteratingMap<2, REAL, DuffingMap<REAL>> iteratingMap_;
  
  __host__ __device__
  inline Duffing(REAL rho = 0.0, REAL eps = 1.0, uint32_t steps = 1)
      : iteratingMap_(DuffingMap<REAL>(rho, eps), steps) {}
  
  __host__ __device__
  inline bool operator()(const REAL * point, REAL * res) const
  {
    return iteratingMap_(point, res);
  }
};

} // namespace b12
