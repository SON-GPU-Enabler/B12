
#pragma once

#include "MapTools.h"


namespace b12 {

template<typename REAL>
struct HenonMap
{
  REAL a_, b_;
  
  __host__ __device__
  inline HenonMap(REAL a, REAL b) : a_(a), b_(b) {}
  
  __host__ __device__
  inline void operator()(const REAL * point, REAL * res) const
  {
    res[0] = REAL(1.0) - a_ * point[0] * point[0] + point[1];
    res[1] = b_ * point[0];
  }
};


template<Dimension DIM, typename REAL>
struct Henon : public thrust::binary_function<REAL*, REAL*, bool>
{
  IteratingMap<2, REAL, HenonMap<REAL>> iteratingMap_;
  
  __host__ __device__
  inline Henon(REAL a = 1.4, REAL b = 0.3, uint32_t steps = 1)
      : iteratingMap_(HenonMap<REAL>(a, b), steps) {}
  
  __host__ __device__
  inline bool operator()(const REAL * point, REAL * res) const
  {
    return iteratingMap_(point, res);
  }
};

} // namespace b12
