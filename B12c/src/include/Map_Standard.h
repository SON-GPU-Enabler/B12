
#pragma once

#include "MapTools.h"


namespace b12 {

// Note:
// remainder[f](x,2*pi)       gives periodic values w.r.t. (-pi,  pi)
// remainder[f](x-pi,2*pi)+pi gives periodic values w.r.t. (  0,2*pi)

template<Dimension DIM, typename REAL>
struct StandardMap;

template<>
struct StandardMap<2, float>
{
  float K_;
  
  __host__ __device__
  inline StandardMap(float K) : K_(K) {}
  
  __host__ __device__
  inline void operator()(const float * point, float * res) const
  {
    // res[0] should be (periodically) in [0,2pi); res[1] in [-pi,pi)
    res[0] = remainderf(point[0] + point[1] - float(M_PI), float(2.0*M_PI)) + float(M_PI);
    res[1] = remainderf(point[1] + K_ * sinf(point[0] + point[1]), float(2.0*M_PI));
  }
};

template<>
struct StandardMap<2, double>
{
  double K_;
  
  __host__ __device__
  inline StandardMap(double K) : K_(K) {}
  
  __host__ __device__
  inline void operator()(const double * point, double * res) const
  {
    // res[0] should be (periodically) in [0,2pi); res[1] in [-pi,pi)
    res[0] = remainder(point[0] + point[1] - double(M_PI), double(2.0*M_PI)) + double(M_PI);
    res[1] = remainder(point[1] + K_ * sin(point[0] + point[1]), double(2.0*M_PI));
  }
};

template<>
struct StandardMap<4, float>
{
  float K1_, K2_, K_;
  
  __host__ __device__
  inline StandardMap(float K1, float K2, float K) : K1_(K1), K2_(K2), K_(K) {}
  
  __host__ __device__
  inline void operator()(const float * point, float * res) const
  {
    // res[0] and res[2] should be (periodically) in [0,2pi); res[1] and res[3] in [-pi,pi)
    res[0] = remainderf(point[0] + point[1] - float(M_PI), float(2.0*M_PI)) + float(M_PI);
    res[2] = remainderf(point[2] + point[3] - float(M_PI), float(2.0*M_PI)) + float(M_PI);
    res[1] = remainderf(point[1] + K1_ * sinf(point[0] + point[1]) + 
                        K_ * sinf(point[0] + point[1] + point[2] + point[3]), float(2.0*M_PI));
    res[3] = remainderf(point[3] + K2_ * sinf(point[2] + point[3]) +
                        K_ * sinf(point[0] + point[1] + point[2] + point[3]), float(2.0*M_PI));
  }
};

template<>
struct StandardMap<4, double>
{
  double K1_, K2_, K_;
  
  __host__ __device__
  inline StandardMap(double K1, double K2, double K) : K1_(K1), K2_(K2), K_(K) {}
  
  __host__ __device__
  inline void operator()(const double * point, double * res) const
  {
    // res[0] and res[2] should be (periodically) in [0,2pi); res[1] and res[3] in [-pi,pi)
    res[0] = remainder(point[0] + point[1] - double(M_PI), double(2.0*M_PI)) + double(M_PI);
    res[2] = remainder(point[2] + point[3] - double(M_PI), double(2.0*M_PI)) + double(M_PI);
    res[1] = remainder(point[1] + K1_ * sin(point[0] + point[1]) + 
                       K_ * sin(point[0] + point[1] + point[2] + point[3]), double(2.0*M_PI));
    res[3] = remainder(point[3] + K2_ * sin(point[2] + point[3]) +
                       K_ * sin(point[0] + point[1] + point[2] + point[3]), double(2.0*M_PI));
  }
};


template<Dimension DIM, typename REAL>
struct Standard;

template<typename REAL>
struct Standard<2, REAL> : public thrust::binary_function<REAL*, REAL*, bool>
{
  IteratingMap<2, REAL, StandardMap<2, REAL>> iteratingMap_;
  
  __host__ __device__
  inline Standard(REAL K, uint32_t steps = 1)
      : iteratingMap_(StandardMap<2, REAL>(K), steps) {}
  
  __host__ __device__
  inline bool operator()(const REAL * point, REAL * res) const
  {
    return iteratingMap_(point, res);
  }
};

template<typename REAL>
struct Standard<4, REAL> : public thrust::binary_function<REAL*, REAL*, bool>
{
  IteratingMap<4, REAL, StandardMap<4, REAL>> iteratingMap_;
  
  __host__ __device__
  inline Standard(REAL K1, REAL K2, REAL K, uint32_t steps = 1) 
      : iteratingMap_(StandardMap<4, REAL>(K1, K2, K), steps) {}
  
  __host__ __device__
  inline bool operator()(const REAL * point, REAL * res) const
  {
    return iteratingMap_(point, res);
  }
};

} // namespace b12
