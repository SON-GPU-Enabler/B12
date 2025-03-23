
#pragma once

#include "MapTools.h"


namespace b12 {

// Functor for the right-hand side of the unforced VanDerPol ODE system
template<typename REAL>
struct VanDerPolRHS
{
  REAL epsilon_;
  
  __host__ __device__
  inline VanDerPolRHS(REAL epsilon)
      : epsilon_(epsilon) {}
  
  __host__ __device__
  inline void operator()(const REAL * point, REAL * res) const
  {
    res[0] = point[1];
    res[1] = epsilon_ * (REAL(1.0) - point[0] * point[0]) * point[1] - point[0];
  }
};


// Functor for the right-hand side of the forced VanDerPol ODE system
template<typename REAL>
struct ForcedVanDerPolRHS;

// additional float version since there is a sinf (sin for float)
template<>
struct ForcedVanDerPolRHS<float>
{
  float epsilon_, F_, omega_;
  
  __host__ __device__
  inline ForcedVanDerPolRHS(float epsilon, float F, float omega)
      : epsilon_(epsilon), F_(F), omega_(omega) {}
  
  __host__ __device__
  inline void operator()(const float * point, float * res) const
  {
    res[0] = point[1];
    res[1] = epsilon_ * (float(1.0) - point[0] * point[0]) * point[1] - 
             point[0] - F_ * sinf(omega_ * point[2]);
    res[2] = float(1.0);
  }
};

template<>
struct ForcedVanDerPolRHS<double>
{
  double epsilon_, F_, omega_;
  
  __host__ __device__
  inline ForcedVanDerPolRHS(double epsilon, double F, double omega)
      : epsilon_(epsilon), F_(F), omega_(omega) {}
  
  __host__ __device__
  inline void operator()(const double * point, double * res) const
  {
    res[0] = point[1];
    res[1] = epsilon_ * (double(1.0) - point[0] * point[0]) * point[1] - 
             point[0] - F_ * sin(omega_ * point[2]);
    res[2] = double(1.0);
  }
};

template<Dimension DIM, typename REAL>
struct VanDerPol;

// Functor wrapping VanDerPol' right-hand side and an integrator
template<typename REAL>
struct VanDerPol<2, REAL> : public thrust::binary_function<REAL*, REAL*, bool>
{
  RK4Auto<2, REAL, VanDerPolRHS<REAL>> integrator_;
  
  __host__ __device__
  inline VanDerPol(REAL epsilon = 5.0, REAL h = 0.01, uint32_t steps = 20)
      : integrator_(VanDerPolRHS<REAL>(epsilon), h, steps) {}
  
  __host__ __device__
  inline bool operator()(const REAL * point, REAL * res) const
  {
    return integrator_(point, res);
  }
};

// Functor wrapping VanDerPol' right-hand side and an integrator
template<typename REAL>
struct VanDerPol<3, REAL> : public thrust::binary_function<REAL*, REAL*, bool>
{
  RK4Auto<3, REAL, ForcedVanDerPolRHS<REAL>> integrator_;
  
  __host__ __device__
  inline VanDerPol(REAL epsilon = 5.0, REAL F = 0.5, REAL omega = 1.15,
                   REAL h = 0.01, uint32_t steps = 20)
      : integrator_(ForcedVanDerPolRHS<REAL>(epsilon, F, omega), h, steps) {}
  
  __host__ __device__
  inline bool operator()(const REAL * point, REAL * res) const
  {
    return integrator_(point, res);
  }
};

} // namespace b12
