
#pragma once

#include "MapTools.h"


namespace b12 {

template<typename REAL>
struct IkedaMap;

template<>
struct IkedaMap<float>
{
  float R_, C1_, C2_, C3_;
  
  __host__ __device__
  inline IkedaMap(float R, float C1, float C2, float C3) : R_(R), C1_(C1), C2_(C2), C3_(C3) {}
  
  __host__ __device__
  inline void operator()(const float * point, float * res) const
  {
    float sinVal, cosVal;
    sincosf(C1_ - C3_ / (point[0] * point[0] + point[1] * point[1] + float(1.0)), &sinVal, &cosVal);
    res[0] = R_ + C2_ * (point[0] * cosVal - point[1] * sinVal);
    res[1] = C2_ * (point[0] * sinVal + point[1] * cosVal);
  }
};

template<>
struct IkedaMap<double>
{
  double R_, C1_, C2_, C3_;
  
  __host__ __device__
  inline IkedaMap(double R, double C1, double C2, double C3) : R_(R), C1_(C1), C2_(C2), C3_(C3) {}
  
  __host__ __device__
  inline void operator()(const double * point, double * res) const
  {
    double sinVal, cosVal;
    sincos(C1_ - C3_ / (point[0] * point[0] + point[1] * point[1] + double(1.0)), &sinVal, &cosVal);
    res[0] = R_ + C2_ * (point[0] * cosVal - point[1] * sinVal);
    res[1] = C2_ * (point[0] * sinVal + point[1] * cosVal);
  }
};


template<Dimension DIM, typename REAL>
struct Ikeda : public thrust::binary_function<REAL*, REAL*, bool>
{
  IteratingMap<2, REAL, IkedaMap<REAL>> iteratingMap_;
  
  __host__ __device__
  inline Ikeda(REAL R = 0.9, REAL C1 = 0.4, REAL C2 = 0.9, REAL C3 = 6.0, uint32_t steps = 1)
      : iteratingMap_(IkedaMap<REAL>(R, C1, C2, C3), steps) {}
  
  __host__ __device__
  inline bool operator()(const REAL * point, REAL * res) const
  {
    return iteratingMap_(point, res);
  }
};

} // namespace b12
