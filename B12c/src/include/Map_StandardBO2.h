
#pragma once

#include "MapTools.h"


namespace b12 {

// Note:
// remainder[f](x,2*pi)       gives periodic values w.r.t. (-pi,  pi)
// remainder[f](x-pi,2*pi)+pi gives periodic values w.r.t. (  0,2*pi)

template<Dimension DIM, typename REAL>
struct StandardBO2Map;

template<>
struct StandardBO2Map<2, float>
{
  float K_, expmlambda_;
  
  __host__ __device__
  inline StandardBO2Map(float K, float lambda) : K_(K), expmlambda_(expf(-lambda)) {}
  
  __host__ __device__
  inline void operator()(const float * point, float * res) const
  {
    res[1] = remainderf(point[1] + K_ / (2.0f * float(M_PI)) * sinpif(2.0f * point[0]), 1.0f);
    res[0] = remainderf(point[0] + res[1] - 0.5f, 1.0f) + 0.5f;
  }
  
  __host__ __device__
  inline void timesJacobian(const float * point, const float * multiplicand, float * res) const
  {
    res[1] = expmlambda_ * (K_ * cospif(2.0f * point[0]) * multiplicand[0] + multiplicand[1]);
    res[0] = res[1] + expmlambda_ * multiplicand[0];
  }
};

template<>
struct StandardBO2Map<2, double>
{
  double K_, expmlambda_;
  
  __host__ __device__
  inline StandardBO2Map(double K, double lambda) : K_(K), expmlambda_(exp(-lambda)) {}
  
  __host__ __device__
  inline void operator()(const double * point, double * res) const
  {
    res[1] = remainder(point[1] + K_ / (2.0 * double(M_PI)) * sinpi(2.0 * point[0]), 1.0);
    res[0] = remainder(point[0] + res[1] - 0.5, 1.0) + 0.5;
  }
  
  __host__ __device__
  inline void timesJacobian(const double * point, const double * multiplicand, double * res) const
  {
    res[1] = expmlambda_ * (K_ * cospi(2.0 * point[0]) * multiplicand[0] + multiplicand[1]);
    res[0] = res[1] + expmlambda_ * multiplicand[0];
  }
};

template<typename REAL>
struct StandardBO2Map<4, REAL>
{
  StandardBO2Map<2, REAL> map_;
  
  __host__ __device__
  inline StandardBO2Map(REAL K, REAL lambda) : map_(K, lambda) {}
  
  __host__ __device__
  inline void operator()(const REAL * point, REAL * res) const
  {
    map_(point, res);
    res[0] += point[2];
    res[1] += point[3];
    map_.timesJacobian(point, point + 2, res + 2);
  }
};


template<Dimension DIM, typename REAL>
struct StandardBO2;

template<typename REAL>
struct StandardBO2<2, REAL> : public thrust::binary_function<REAL*, REAL*, bool>
{
  IteratingMap<2, REAL, StandardBO2Map<2, REAL>> iteratingMap_;
  
  __host__ __device__
  inline StandardBO2(REAL K, uint32_t steps = 1) 
      : iteratingMap_(StandardBO2Map<2, REAL>(K, REAL(0.0)), steps) {}
  
  __host__ __device__
  inline bool operator()(const REAL * point, REAL * res) const
  {
    return iteratingMap_(point, res);
  }
};

template<typename REAL>
struct StandardBO2<4, REAL> : public thrust::binary_function<REAL*, REAL*, bool>
{
  IteratingMap<4, REAL, StandardBO2Map<4, REAL>> iteratingMap_;
  
  __host__ __device__
  inline StandardBO2(REAL K, REAL lambda, uint32_t steps = 1) 
      : iteratingMap_(StandardBO2Map<4, REAL>(K, lambda), steps) {}
  
  __host__ __device__
  inline bool operator()(const REAL * point, REAL * res) const
  {
    return iteratingMap_(point, res);
  }
};

} // namespace b12
