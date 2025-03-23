
#pragma once

#include "MapTools.h"


namespace b12 {

// Functor for the right-hand side of the Moussa ODE system
template<typename REAL>
struct MoussaRHS;

// Functor for the right-hand side of the Moussa ODE system
template<>
struct MoussaRHS<float>
{
  __host__ __device__
  inline void operator()(float t, const float * point, float * res) const
  {
    t = t * t * (3.0f - 2.0f * t);
    
    float s0, c0, s1, c1;
    
    sincospif(point[0], & s0, & c0);
    sincospif(point[1], & s1, & c1);
    
    // sinpif(2 * x) == 2 * sinpif(x) * cospif(x)
    // cospif(2 * x) == 1 - 2 * sinpif(x) * sinpif(x)
    
    res[0] = (-float(M_PI) * t * 2.0f * s0 * c0 - (1.0f - t) * float(M_PI) * s0) * c1;
    res[1] = (2.0f * float(M_PI) * t * (1.0f - 2.0f * s0 * s0) + (1.0f - t) * float(M_PI) * c0) * s1; 
    
//     res[0] = (-float(M_PI) * t * sinpif(2.0f * point[0]) - (1.0f - t) * float(M_PI) * sinpif(point[0])) * cospif(point[1]);
//     res[1] = (2.0f * float(M_PI) * t * cospif(2.0f * point[0]) + (1.0f - t) * float(M_PI) * cospif(point[0])) * sinpif(point[1]); 
  }
};

// Functor for the right-hand side of the Moussa ODE system
template<>
struct MoussaRHS<double>
{
  __host__ __device__
  inline void operator()(double t, const double * point, double * res) const
  {
    t = t * t * (3.0 - 2.0 * t);
    
    double s0, c0, s1, c1;
    
    sincospi(point[0], & s0, & c0);
    sincospi(point[1], & s1, & c1);
    
    // sinpi(2 * x) == 2 * sinpi(x) * cospi(x)
    // cospi(2 * x) == 1 - 2 * sinpi(x) * sinpi(x)
    
    res[0] = (-double(M_PI) * t * 2.0 * s0 * c0 - (1.0 - t) * double(M_PI) * s0) * c1;
    res[1] = (2.0 * double(M_PI) * t * (1.0 - 2.0 * s0 * s0) + (1.0 - t) * double(M_PI) * c0) * s1; 
  }
};

// Functor wrapping Moussa' right-hand side and an integrator
template<Dimension DIM, typename REAL>
struct Moussa : public thrust::binary_function<REAL*, REAL*, bool>
{
  RK4<2, REAL, MoussaRHS<REAL>> integrator_;
  
  __host__ __device__
  inline Moussa(REAL t0 = 0.0, REAL tEnd = 1.0, REAL h = 0.001)
      : integrator_(MoussaRHS<REAL>(), t0, tEnd, h) {}
  
  __host__ __device__
  inline bool operator()(const REAL * point, REAL * res) const
  {
    return integrator_(point, res);
  }
};

} // namespace b12
