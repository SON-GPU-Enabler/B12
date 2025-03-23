
#pragma once

#include "MapTools.h"


namespace b12 {

// Functor for the right-hand side of the Dahlquist ODE system
template<typename REAL>
struct DahlquistRHS;

// Functor for the right-hand side of the Dahlquist ODE system
template<>
struct DahlquistRHS<float>
{
  __host__ __device__
  inline void operator()(float t, const float * point, float * res) const
  {
    float f = t * t * (3.0f - 2.0f * t);
    
    res[0] = (-float(M_PI) * f * sinpif(2.0f * point[0]) - (1.0f - f) * float(M_PI) * sinpif(point[0])) * cospif(point[1]);
    res[1] = (2.0f * float(M_PI) * f * cospif(2.0f * point[0]) + (1.0f - f) * float(M_PI) * cospif(point[0])) * sinpif(point[1]); 
  }
};

// Functor for the right-hand side of the Dahlquist ODE system
template<>
struct DahlquistRHS<double>
{
  __host__ __device__
  inline void operator()(double t, const double * point, double * res) const
  {
    double f = t * t * (3.0 - 2.0 * t);
    
    res[0] = (-double(M_PI) * f * sinpi(2.0 * point[0]) - (1.0 - f) * double(M_PI) * sinpi(point[0])) * cospi(point[1]);
    res[1] = (2.0 * double(M_PI) * f * cospi(2.0 * point[0]) + (1.0 - f) * double(M_PI) * cospi(point[0])) * sinpi(point[1]); 
  }
};

// Functor wrapping Dahlquist' right-hand side and an integrator
template<Dimension DIM, typename REAL>
struct Dahlquist : public thrust::binary_function<REAL*, REAL*, bool>
{
  RK4<2, REAL, DahlquistRHS<REAL>> integrator_;
  
  __host__ __device__
  inline Dahlquist(REAL t0 = 0.0, REAL tEnd = 1.0, REAL h = 0.01)
      : integrator_(DahlquistRHS<REAL>(), t0, tEnd, h) {}
  
  __host__ __device__
  inline bool operator()(const REAL * point, REAL * res) const
  {
    return integrator_(point, res);
  }
};

} // namespace b12
