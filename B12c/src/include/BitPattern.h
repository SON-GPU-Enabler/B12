
#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include "ExtendedUInt64_t.h"
#include "TypeDefinitions.h"


namespace b12 {

// N must be the exact number of stored bits!!!

template<Depth N>
struct UIntWrapper
{
  using Type = void;
};

template<>
struct UIntWrapper<32>
{
  using Type = uint32_t;
};

template<>
struct UIntWrapper<64>
{
  using Type = uint64_t;
};

template<>
struct UIntWrapper<72>
{
  using Type = ExtendedUInt64_t<uint8_t>;
};

template<>
struct UIntWrapper<80>
{
  using Type = ExtendedUInt64_t<uint16_t>;
};

template<>
struct UIntWrapper<96>
{
  using Type = ExtendedUInt64_t<uint32_t>;
};

template<>
struct UIntWrapper<128>
{
  using Type = ExtendedUInt64_t<uint64_t>;
};


template<Depth N>
using BitPattern = typename UIntWrapper<N>::Type;


// determine maximum of MaxNBits_aux<"lower half"> and MaxNBits_aux<"upper half">
template<Depth L, Depth U>
struct MaxNBits_aux
{
  const static Depth value = MaxNBits_aux<U/2 + L/2 + 1, U>::value > MaxNBits_aux<L, U/2 + L/2>::value ?
                             MaxNBits_aux<U/2 + L/2 + 1, U>::value :
                             MaxNBits_aux<L, U/2 + L/2>::value;
};

// if BitPattern<N> is only void, value = 0, else value = N
template<Depth N>
struct MaxNBits_aux<N, N>
{
  const static Depth value = std::is_void<BitPattern<N>>::value ? 0 : N;
};

// determine "greatest" available N as BitPattern length
struct MaxNBits
{
  const static Depth value = MaxNBits_aux<std::numeric_limits<Depth>::min(), std::numeric_limits<Depth>::max()>::value;
};


// if i <= 0, b is returned;
// if i >= N, 0 is returned;
// otherwise b shifted by i
template<Depth N>
__host__ __device__
BitPattern<N> safeLeftShift(const BitPattern<N>& b, int i);

template<Depth N>
__host__ __device__
BitPattern<N> safeRightShift(const BitPattern<N>& b, int i);

} // namespace b12


#include "BitPattern.hpp"
