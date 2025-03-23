
#pragma once

#include "MapTools.h"


namespace b12 {

// Functors for the right-hand side of Hooke's Law:
// x' = v, v' = -k/m * x
template<typename REAL>
struct HookesRHS1
{
  __host__ __device__
  inline void operator()(const REAL * point, REAL * res) const
  {
    res[0] = point[0];
  }
};

template<typename REAL>
struct HookesRHS2
{
  REAL k_, m_;
  
  __host__ __device__
  inline HookesRHS2(REAL k, REAL m) : k_(k), m_(m) {}
  
  __host__ __device__
  inline void operator()(const REAL * point, REAL * res) const
  {
    res[0] = -k_ / m_ * point[0];
  }
};

// Functor wrapping Lorenz' right-hand side and an integrator
template<Dimension DIM, typename REAL>
struct HookesLaw : public thrust::binary_function<REAL*, REAL*, bool>
{
  SymplecticIntegratorAuto<2, REAL, HookesRHS1<REAL>, HookesRHS2<REAL>> integrator_;
  
  __host__ __device__
  inline HookesLaw(REAL k, REAL m,
                   uint8_t order = 3, REAL h = 0.01, uint32_t steps = 100)
      : integrator_(order, HookesRHS1<REAL>(), HookesRHS2<REAL>(k, m), h, steps) {}
  
  __host__ __device__
  inline bool operator()(const REAL * point, REAL * res) const
  {
    return integrator_(point, res);
  }
};

} // namespace b12
