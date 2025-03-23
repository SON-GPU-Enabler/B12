
#pragma once

#include <cstdint>

#include <thrust/functional.h>

#include "TypeDefinitions.h"


namespace b12 {

// Functor that applies a map iteratively
template<Dimension DIM, typename REAL, typename MAP>
struct IteratingMap : public thrust::binary_function<REAL *, REAL *, bool>
{
  MAP map_;
  uint32_t steps_;
  
  __host__ __device__
  IteratingMap(MAP map, uint32_t steps);
  
  __host__ __device__
  bool operator()(const REAL * point, REAL * res) const;
};


// Functor for the classical Runge-Kutta method
template<Dimension DIM, typename REAL, typename RHS>
struct RK4 : public thrust::binary_function<REAL *, REAL *, bool>
{
  RHS rightHandSide_;
  REAL t0_, tEnd_, h_;
  
  __host__ __device__
  RK4(RHS rightHandSide, REAL t0, REAL tEnd, REAL h);
  
  __host__ __device__
  bool operator()(const REAL * point, REAL * res) const;
};


// Functor for the (autonomous) classical Runge-Kutta method
template<Dimension DIM, typename REAL, typename RHS>
struct RK4Auto : public thrust::binary_function<REAL *, REAL *, bool>
{
  RHS rightHandSide_;
  REAL h_;
  uint32_t steps_;
  
  __host__ __device__
  RK4Auto(RHS rightHandSide, REAL h, uint32_t steps);
  
  __host__ __device__
  bool operator()(const REAL * point, REAL * res) const;
};


// Functor for symplectic integrator methods (with order 1, 2, 3, or 4)
// solving y1' = RHS1(t, y2), y2' = RHS2(t, y1), with y1, y2 \in R^(DIM/2)
// where y1 stands for position and y2 for velocity
template<Dimension DIM, typename REAL, typename RHS1, typename RHS2>
struct SymplecticIntegrator : public thrust::binary_function<REAL *, REAL *, bool>
{
  uint8_t order_;
  RHS1 rightHandSide1_;
  RHS2 rightHandSide2_;
  REAL t0_, tEnd_, h_;
  
  __host__ __device__
  SymplecticIntegrator(uint8_t order, RHS1 rightHandSide1, RHS2 rightHandSide2, REAL t0, REAL tEnd, REAL h);
  
  __host__ __device__
  bool operator()(const REAL * point, REAL * res) const;
};


// Functor for symplectic integrator methods (with order 1, 2, 3, or 4)
// solving y1' = RHS1(y2), y2' = RHS2(y1), with y1, y2 \in R^(DIM/2)
// where y1 stands for position and y2 for velocity
template<Dimension DIM, typename REAL, typename RHS1, typename RHS2>
struct SymplecticIntegratorAuto : public thrust::binary_function<REAL *, REAL *, bool>
{
  uint8_t order_;
  RHS1 rightHandSide1_;
  RHS2 rightHandSide2_;
  REAL h_;
  uint32_t steps_;
  
  __host__ __device__
  SymplecticIntegratorAuto(uint8_t order, RHS1 rightHandSide1, RHS2 rightHandSide2, REAL h, uint32_t steps);
  
  __host__ __device__
  bool operator()(const REAL * point, REAL * res) const;
};

} // namespace b12


#include "MapTools.hpp"
