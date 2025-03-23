
#pragma once

#include "MapTools.h"


namespace b12 {

template<Dimension DIM, typename REAL>
struct Identity : public thrust::binary_function<REAL*, REAL*, bool>
{
  __host__ __device__
  inline bool operator()(const REAL * point, REAL * res) const
  {
#pragma unroll
    for (Dimension i = 0; i < DIM; ++i) {
      res[i] = point[i]; // indices are static/known at compile time due to loop unrolling
    }
    
    return false;
  }
};

} // namespace b12
