
#include "math.h"


namespace b12 {

template<Dimension EXP, typename T>
__host__ __device__
inline T ipow(T base)
{
  T res(1);
#pragma unroll
  for (Dimension i = EXP; i > Dimension(0); i >>= 1) {
    if (i & Dimension(1)) {
      res *= base;
    }
    base *= base;
  }
  return res;
}

__host__ __device__
inline int countLeadingZeros(uint32_t ui)
{
#ifdef __CUDA_ARCH__
  // cuda version for 32 bit
  return __clz(ui);
#else
  // gcc version for unsigned long long; case __builtin_clz(0) has undefined behaviour
  return ui ? __builtin_clz(ui) : 32;
#endif
}
  
__host__ __device__
inline int countLeadingZeros(uint64_t ui)
{
#ifdef __CUDA_ARCH__
  // cuda version for 64 bit
  return __clzll(ui);
#else
  // gcc version for unsigned long long; case __builtin_clz(0) has undefined behaviour
  return ui ? __builtin_clzll(ui) : 64;
#endif
}

__host__ __device__
inline float floorreal(float arg)
{
  return floorf(arg);
}

__host__ __device__
inline double floorreal(double arg)
{
  return floor(arg);
}

__host__ __device__
inline float ldexpreal(float arg, int exp)
{
  return ldexpf(arg, exp);
}

__host__ __device__
inline double ldexpreal(double arg, int exp)
{
  return ldexp(arg, exp);
}

__host__ __device__
inline float cbrtreal(float arg)
{
  return cbrtf(arg);
}

__host__ __device__
inline double cbrtreal(double arg)
{
  return cbrt(arg);
}

template<>
__device__
inline float curand_uniform_real<float>(curandStatePhilox4_32_10_t* state, float a, float b)
{
  // curand_uniform generates a float in (0, 1]
  return -curand_uniform(state) * (b - a) + b;
}

template<>
__device__
inline double curand_uniform_real<double>(curandStatePhilox4_32_10_t* state, double a, double b)
{
  // curand_uniform_double generates a double in (0, 1]
  return -curand_uniform_double(state) * (b - a) + b;
}

inline std::ostream& operator<<(std::ostream& os, const uint8_t& ui)
{
  return os << uint32_t(ui);
}

inline std::istream& operator>>(std::istream& is, uint8_t& ui)
{
  uint32_t temp;
  is >> temp;
  ui = temp;
  return is;
}

} // namespace b12
