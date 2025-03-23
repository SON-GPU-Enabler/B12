
#pragma once

#include "MapTools.h"


namespace b12 {

// Functor for the right-hand side of the Lorenz ODE system
template<typename REAL>
struct LorenzRHS
{
  REAL sigma_, rho_, beta_;
  
  __host__ __device__
  inline LorenzRHS(REAL sigma, REAL rho, REAL beta)
      : sigma_(sigma), rho_(rho), beta_(beta) {}
  
  __host__ __device__
  inline void operator()(const REAL * point, REAL * res) const
  {
    res[0] = sigma_ * (point[1] - point[0]);
    res[1] = rho_ * point[0] - point[1] - point[0] * point[2];
    res[2] = -beta_ * point[2] + point[0] * point[1];
  }
};

// Functor wrapping Lorenz' right-hand side and an integrator
template<Dimension DIM, typename REAL>
struct Lorenz : public thrust::binary_function<REAL*, REAL*, bool>
{
  RK4Auto<3, REAL, LorenzRHS<REAL>> integrator_;
  
  __host__ __device__
  inline Lorenz(REAL sigma = 10.0, REAL rho = 28.0, REAL beta = 8.0/3.0,
                REAL h = 0.01, uint32_t steps = 20)
      : integrator_(LorenzRHS<REAL>(sigma, rho, beta), h, steps) {}
  
  __host__ __device__
  inline bool operator()(const REAL * point, REAL * res) const
  {
    return integrator_(point, res);
  }
};

} // namespace b12
