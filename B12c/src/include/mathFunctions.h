
#pragma once

#include <cstdint>
#include <iostream>

#include <curand_kernel.h>

#include "TypeDefinitions.h"


namespace b12 {

// power function when the integer exponent is known at compile time
template<Dimension EXP, typename T>
__host__ __device__
T ipow(T base);

__host__ __device__
int countLeadingZeros(uint32_t ui);

__host__ __device__
int countLeadingZeros(uint64_t ui);

__host__ __device__
float floorreal(float arg);

__host__ __device__
double floorreal(double arg);

__host__ __device__
float ldexpreal(float arg, int exp);

__host__ __device__
double ldexpreal(double arg, int exp);

__host__ __device__
float cbrtreal(float arg);

__host__ __device__
double cbrtreal(double arg);

// generates a random number uniformly distributed in [a, b)
template<typename REAL>
__device__
REAL curand_uniform_real(curandStatePhilox4_32_10_t* state, REAL a, REAL b);

std::ostream& operator<<(std::ostream& os, const uint8_t& ui);

std::istream& operator>>(std::istream& is, uint8_t& ui);

} // namespace b12


#include "mathFunctions.hpp"
