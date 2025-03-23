
#pragma once

#include <thrust/functional.h>

#include "TypeDefinitions.h"


namespace b12 {

// functor representing a grid of the surface of a hypercube, each face represented by a (DIM-1)-dimensional FullGrid;
// points may be traversed multiple times
template<Dimension DIM, typename REAL>
struct FaceGrid : public thrust::binary_function<NrPoints, REAL*, bool>
{
  NrPoints nPointsPerDim_; // nPointsPerDim_ ^ (DIM - 1) = number of points per (DIM - 1)-dimensional face
  REAL eps_;
  
  __host__ __device__
  FaceGrid(NrPoints nPointsPerDim = 2, REAL eps = 0.0); // default: vertices of the box
  
  __host__
  FaceGrid(NrPoints nPointsPerDim = 2, NrPoints factorEps = 0); // eps_ = factorEps * std::numeric_limits<REAL>::epsilon()
  
  __host__ __device__
  bool operator()(NrPoints i, REAL * res) const;
  
  __host__ __device__
  Dimension dim() const;
  
  __host__ __device__
  NrPoints getNumberOfPointsPerBox() const;
};

// functor representing an equidistant grid of ([-1,1]^DIM)*(1+eps_) including the edges
template<Dimension DIM, typename REAL>
struct FullGrid : public thrust::binary_function<NrPoints, REAL*, bool>
{
  NrPoints nPointsPerDim_; // number of points per dimension
  REAL eps_;
  
  __host__ __device__
  FullGrid(NrPoints nPointsPerDim = 2, REAL eps = 0.0); // default: vertices of the box
  
  __host__
  FullGrid(NrPoints nPointsPerDim = 2, NrPoints factorEps = 0); // eps_ = factorEps * std::numeric_limits<REAL>::epsilon()
  
  __host__ __device__
  bool operator()(NrPoints i, REAL * res) const;
  
  __host__ __device__
  Dimension dim() const;
  
  __host__ __device__
  NrPoints getNumberOfPointsPerBox() const;
};

// functor representing an equidistant grid of [-1,1]^DIM such that "edge" 
// points of neighbouring boxes have the same distance like inside a box
template<Dimension DIM, typename REAL>
struct InnerGrid : public thrust::binary_function<NrPoints, REAL*, bool>
{
  NrPoints nPointsPerDim_; // number of points per dimension
  
  __host__ __device__
  InnerGrid(NrPoints nPointsPerDim = 1); // default: center point of the box
  
  __host__ __device__
  bool operator()(NrPoints i, REAL * res) const;
  
  __host__ __device__
  Dimension dim() const;
  
  __host__ __device__
  NrPoints getNumberOfPointsPerBox() const;
};

// functor representing a random, uniformly distributed grid of [-1,1)^DIM;
// on the host, mt19937 is used as random number generator where each thread gets seed_ + i as seed;
// on the device, Philox_4x32_10 is used, each thread has the same seed and starts with the (i*2^64)-th random number
template<Dimension DIM, typename REAL>
struct UniformGrid : public thrust::binary_function<NrPoints, REAL*, bool>
{
  NrPoints nPoints_; // number of points
  unsigned long long seed_; // seed for the random generator
  
  __host__ __device__
  UniformGrid(NrPoints nPoints = 1, unsigned long long seed = 0); // default: one random point of the box
  
  __host__ __device__
  bool operator()(NrPoints i, REAL * res) const;
  
  __host__ __device__
  Dimension dim() const;
  
  __host__ __device__
  NrPoints getNumberOfPointsPerBox() const;
};

} // namespace b12


#include "Grids.hpp"
