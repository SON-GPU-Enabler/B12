
#pragma once

#include "MapTools.h"


namespace b12 {

// Note:
// remainder[f](x,2*pi)       gives periodic values w.r.t. (-pi,  pi)
// remainder[f](x-pi,2*pi)+pi gives periodic values w.r.t. (  0,2*pi)

template<Dimension DIM, typename REAL>
struct StandardOpenMap;

template<>
struct StandardOpenMap<2, float>
{
  float K_;
  
  __host__ __device__
  inline StandardOpenMap(float K) : K_(K) {}
  
  __host__ __device__
  inline void operator()(const float * point, float * res) const
  {
    // res[0] should be (periodically) in [0,2pi); res[1] in [-pi,pi)
    res[0] = point[0] + point[1];
    res[1] = point[1] + K_ * sinf(point[0] + point[1]);
  }
};

template<>
struct StandardOpenMap<2, double>
{
  double K_;
  
  __host__ __device__
  inline StandardOpenMap(double K) : K_(K) {}
  
  __host__ __device__
  inline void operator()(const double * point, double * res) const
  {
    // res[0] should be (periodically) in [0,2pi); res[1] in [-pi,pi)
    res[0] = point[0] + point[1];
    res[1] = point[1] + K_ * sin(point[0] + point[1]);
  }
};

template<>
struct StandardOpenMap<4, float>
{
  float K1_, K2_, K_;
  
  __host__ __device__
  inline StandardOpenMap(float K1, float K2, float K) : K1_(K1), K2_(K2), K_(K) {}
  
  __host__ __device__
  inline void operator()(const float * point, float * res) const
  {
    // res[0] and res[2] should be (periodically) in [0,2pi); res[1] and res[3] in [-pi,pi)
    res[0] = point[0] + point[1];
    res[2] = point[2] + point[3];
    res[1] = point[1] + K1_ * sinf(point[0] + point[1]) + 
                        K_  * sinf(point[0] + point[1] + point[2] + point[3]);
    res[3] = point[3] + K2_ * sinf(point[2] + point[3]) +
                        K_  * sinf(point[0] + point[1] + point[2] + point[3]);
  }
};

template<>
struct StandardOpenMap<4, double>
{
  double K1_, K2_, K_;
  
  __host__ __device__
  inline StandardOpenMap(double K1, double K2, double K) : K1_(K1), K2_(K2), K_(K) {}
  
  __host__ __device__
  inline void operator()(const double * point, double * res) const
  {
    // res[0] and res[2] should be (periodically) in [0,2pi); res[1] and res[3] in [-pi,pi)
    res[0] = point[0] + point[1];
    res[2] = point[2] + point[3];
    res[1] = point[1] + K1_ * sin(point[0] + point[1]) + 
                        K_  * sin(point[0] + point[1] + point[2] + point[3]);
    res[3] = point[3] + K2_ * sin(point[2] + point[3]) +
                        K_  * sin(point[0] + point[1] + point[2] + point[3]);
  }
};


template<Dimension DIM, typename REAL>
struct StandardOpen;

template<typename REAL>
struct StandardOpen<2, REAL> : public thrust::binary_function<REAL*, REAL*, bool>
{
  IteratingMap<2, REAL, StandardOpenMap<2, REAL>> iteratingMap_;
  
  __host__ __device__
  inline StandardOpen(REAL K, uint32_t steps = 1)
      : iteratingMap_(StandardOpenMap<2, REAL>(K), steps) {}
  
  __host__ __device__
  inline bool operator()(const REAL * point, REAL * res) const
  {
    return iteratingMap_(point, res);
  }
};

template<typename REAL>
struct StandardOpen<4, REAL> : public thrust::binary_function<REAL*, REAL*, bool>
{
  IteratingMap<4, REAL, StandardOpenMap<4, REAL>> iteratingMap_;
  
  __host__ __device__
  inline StandardOpen(REAL K1, REAL K2, REAL K, uint32_t steps = 1) 
      : iteratingMap_(StandardOpenMap<4, REAL>(K1, K2, K), steps) {}
  
  __host__ __device__
  inline bool operator()(const REAL * point, REAL * res) const
  {
    return iteratingMap_(point, res);
  }
};

} // namespace b12
