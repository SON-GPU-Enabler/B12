
#pragma once

#include "MapTools.h"


namespace b12 {

template<typename REAL>
struct TDuffingRHS;

template<>
struct TDuffingRHS<float>
{
  __host__ __device__
  inline void operator()(float t, const float * point, float * res) const
  {
    res[0] = point[1];
    res[1] = (atanf(t) - point[0] * point[0]) * point[0];
  }
};

template<>
struct TDuffingRHS<double>
{
  __host__ __device__
  inline void operator()(double t, const double * point, double * res) const
  {
    res[0] = point[1];
    res[1] = (atan(t) - point[0] * point[0]) * point[0];
  }
};


template<Dimension DIM, typename REAL>
struct TDuffing : public thrust::binary_function<REAL*, REAL*, bool>
{
  RK4<2, REAL, TDuffingRHS<REAL>> integrator_;
  
  __host__ __device__
  inline TDuffing(REAL t0, REAL tEnd, REAL h = 0.001)
      : integrator_(TDuffingRHS<REAL>(), t0, tEnd, h) {}
  
  __host__ __device__
  inline bool operator()(const REAL * point, REAL * res) const
  {
    return integrator_(point, res);
  }
};

} // namespace b12
