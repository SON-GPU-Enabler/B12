
#pragma once

#include "MapTools.h"


namespace b12 {

// Functor for the right-hand side of the Lorenz ODE system
template<typename REAL>
struct ChuaRHS
{
  REAL alpha_, beta_, m0_, m1_;
  
  __host__ __device__
  inline ChuaRHS(REAL alpha, REAL beta, REAL m0, REAL m1)
      : alpha_(alpha), beta_(beta), m0_(m0), m1_(m1) {}
  
  __host__ __device__
  inline void operator()(const REAL * point, REAL * res) const
  {
    res[0] = alpha_ * (point[1] - m0_ * point[0] - m1_ / REAL(3.0) * point[0] * point[0] * point[0]);
    res[1] = point[0] - point[1] + point[2];
    res[2] = -beta_ * point[1];
  }
};

// Functor wrapping Lorenz' right-hand side and an integrator
template<Dimension DIM, typename REAL>
struct Chua : public thrust::binary_function<REAL*, REAL*, bool>
{
  RK4Auto<3, REAL, ChuaRHS<REAL>> integrator_;
  
  __host__ __device__
  inline Chua(REAL alpha = 18.0, REAL beta = 33.0,
              REAL m0 = -0.2, REAL m1 = 0.01,
              REAL h = 0.01, uint32_t steps = 50)
      : integrator_(ChuaRHS<REAL>(alpha, beta, m0, m1), h, steps) {}
  
  __host__ __device__
  inline bool operator()(const REAL * point, REAL * res) const
  {
    return integrator_(point, res);
  }
};

} // namespace b12
