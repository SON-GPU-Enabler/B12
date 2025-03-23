
#pragma once

#include <cstdint>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/cpp/execution_policy.h>
#include <thrust/system/cpp/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/vector.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/system/omp/vector.h>
// #include <thrust/system/tbb/execution_policy.h>
// #include <thrust/system/tbb/vector.h>
#include <thrust/tuple.h>

#include "BitPattern.h"
#include "Flags.h"
#include "TypeDefinitions.h"


namespace b12 {

// this enum allows one implementation (template<Architecture A>) of ImplicitBoxTree
// for both host and device code
enum Architecture {
  HOST,
  DEVICE,
  CPP,
  CUDA,
  OMP,
  TBB
};



// specifies the used memory: either HOST or DEVICE
template<Architecture A>
struct Memory
{
  // checks whether the Architecture uses the host memory or not
  static bool isOnHost();
  
  // for device memory, the current device for the calling host thread is printed and returned;
  // for host memory, nothing is printed and 0 is returned
  static int printDevice();
  
  // computes the available bytes smaller than initFreeBytes
  static uint64_t getFreeBytes(uint64_t initFreeBytes);
};



// dummy class, i.e. only declared for specialisations
template<Architecture A>
struct ThrustSystem;



// template specialisation for host implementations, i.e.
template<>
struct ThrustSystem<HOST>
{
  using Memory = Memory<HOST>;
  
  // typename ThrustSystem<HOST>::execution_policy() has the same effect as thrust::host
  using execution_policy = decltype(thrust::host);
  
  // typename ThrustSystem<HOST>::Vector<T> == thrust::host_vector<T>
  template<typename T>
  using Vector = thrust::host_vector<T>; // container type
  
  template<Depth N>
  using ConstBitPatternIterator = typename thrust::host_vector<BitPattern<N>>::const_iterator;
  using ConstDepthIterator = thrust::host_vector<Depth>::const_iterator;
  using ConstFlagsIterator = thrust::host_vector<Flags>::const_iterator;
  template<Depth N>
  using ConstBoxIterator = thrust::zip_iterator<
                               thrust::tuple<
                                   ConstBitPatternIterator<N>,
                                   ConstDepthIterator,
                                   ConstFlagsIterator>>;
  
  template<Depth N>
  using BitPatternIterator = typename thrust::host_vector<BitPattern<N>>::iterator;
  using DepthIterator = thrust::host_vector<Depth>::iterator;
  using FlagsIterator = thrust::host_vector<Flags>::iterator;
  template<Depth N>
  using BoxIterator = thrust::zip_iterator<
                          thrust::tuple<
                              BitPatternIterator<N>,
                              DepthIterator,
                              FlagsIterator>>;
};



// template specialisation for device implementations, i.e.
template<>
struct ThrustSystem<DEVICE>
{
  using Memory = Memory<DEVICE>;
  
  // typename ThrustSystem<DEVICE>::execution_policy() has the same effect as thrust::device
  using execution_policy = decltype(thrust::device);
  
  // typename ThrustSystem<DEVICE>::Vector<T> == thrust::device_vector<T>
  template<typename T>
  using Vector = thrust::device_vector<T>; // container type
  
  template<Depth N>
  using ConstBitPatternIterator = typename thrust::device_vector<BitPattern<N>>::const_iterator;
  using ConstDepthIterator = thrust::device_vector<Depth>::const_iterator;
  using ConstFlagsIterator = thrust::device_vector<Flags>::const_iterator;
  template<Depth N>
  using ConstBoxIterator = thrust::zip_iterator<
                               thrust::tuple<
                                   ConstBitPatternIterator<N>,
                                   ConstDepthIterator,
                                   ConstFlagsIterator>>;
  
  template<Depth N>
  using BitPatternIterator = typename thrust::device_vector<BitPattern<N>>::iterator;
  using DepthIterator = thrust::device_vector<Depth>::iterator;
  using FlagsIterator = thrust::device_vector<Flags>::iterator;
  template<Depth N>
  using BoxIterator = thrust::zip_iterator<
                          thrust::tuple<
                              BitPatternIterator<N>,
                              DepthIterator,
                              FlagsIterator>>;
};



// template specialisation for Cpp implementations, i.e.
template<>
struct ThrustSystem<CPP>
{
  using Memory = Memory<HOST>;
  
  // typename ThrustSystem<CPP>::execution_policy() has the same effect as thrust::cpp::par
  using execution_policy = decltype(thrust::cpp::par);
  
  // typename ThrustSystem<CPP>::Vector<T> == thrust::host_vector<T>
  template<typename T>
  using Vector = thrust::host_vector<T>; // container type
  
  template<Depth N>
  using ConstBitPatternIterator = typename thrust::host_vector<BitPattern<N>>::const_iterator;
  using ConstDepthIterator = thrust::host_vector<Depth>::const_iterator;
  using ConstFlagsIterator = thrust::host_vector<Flags>::const_iterator;
  template<Depth N>
  using ConstBoxIterator = thrust::zip_iterator<
                               thrust::tuple<
                                   ConstBitPatternIterator<N>,
                                   ConstDepthIterator,
                                   ConstFlagsIterator>>;
  
  template<Depth N>
  using BitPatternIterator = typename thrust::host_vector<BitPattern<N>>::iterator;
  using DepthIterator = thrust::host_vector<Depth>::iterator;
  using FlagsIterator = thrust::host_vector<Flags>::iterator;
  template<Depth N>
  using BoxIterator = thrust::zip_iterator<
                          thrust::tuple<
                              BitPatternIterator<N>,
                              DepthIterator,
                              FlagsIterator>>;
};



// template specialisation for CUDA implementations, i.e.
template<>
struct ThrustSystem<CUDA>
{
  using Memory = Memory<DEVICE>;
  
  // typename ThrustSystem<CUDA>::execution_policy() has the same effect as thrust::cuda::par
  using execution_policy = decltype(thrust::cuda::par);
  
  // typename ThrustSystem<CUDA>::Vector<T> == thrust::device_vector<T>
  template<typename T>
  using Vector = thrust::device_vector<T>; // container type
  
  template<Depth N>
  using ConstBitPatternIterator = typename thrust::device_vector<BitPattern<N>>::const_iterator;
  using ConstDepthIterator = thrust::device_vector<Depth>::const_iterator;
  using ConstFlagsIterator = thrust::device_vector<Flags>::const_iterator;
  template<Depth N>
  using ConstBoxIterator = thrust::zip_iterator<
                               thrust::tuple<
                                   ConstBitPatternIterator<N>,
                                   ConstDepthIterator,
                                   ConstFlagsIterator>>;
  
  template<Depth N>
  using BitPatternIterator = typename thrust::device_vector<BitPattern<N>>::iterator;
  using DepthIterator = thrust::device_vector<Depth>::iterator;
  using FlagsIterator = thrust::device_vector<Flags>::iterator;
  template<Depth N>
  using BoxIterator = thrust::zip_iterator<
                          thrust::tuple<
                              BitPatternIterator<N>,
                              DepthIterator,
                              FlagsIterator>>;
};



// template specialisation for OpenMP implementations, i.e.
template<>
struct ThrustSystem<OMP>
{
  using Memory = Memory<HOST>;
  
  // typename ThrustSystem<OMP>::execution_policy() has the same effect as thrust::omp::par
  using execution_policy = decltype(thrust::omp::par);
  
  // typename ThrustSystem<OMP>::Vector<T> == thrust::host_vector<T>
  template<typename T>
  using Vector = thrust::host_vector<T>; // container type
  
  template<Depth N>
  using ConstBitPatternIterator = typename thrust::host_vector<BitPattern<N>>::const_iterator;
  using ConstDepthIterator = thrust::host_vector<Depth>::const_iterator;
  using ConstFlagsIterator = thrust::host_vector<Flags>::const_iterator;
  template<Depth N>
  using ConstBoxIterator = thrust::zip_iterator<
                               thrust::tuple<
                                   ConstBitPatternIterator<N>,
                                   ConstDepthIterator,
                                   ConstFlagsIterator>>;
  
  template<Depth N>
  using BitPatternIterator = typename thrust::host_vector<BitPattern<N>>::iterator;
  using DepthIterator = thrust::host_vector<Depth>::iterator;
  using FlagsIterator = thrust::host_vector<Flags>::iterator;
  template<Depth N>
  using BoxIterator = thrust::zip_iterator<
                          thrust::tuple<
                              BitPatternIterator<N>,
                              DepthIterator,
                              FlagsIterator>>;
};



// // template specialisation for TBB implementations, i.e.
// template<>
// struct ThrustSystem<TBB>
// {
//   using Memory = Memory<HOST>;
//   
//   // typename ThrustSystem<TBB>::execution_policy() has the same effect as thrust::tbb::par
//   using execution_policy = decltype(thrust::tbb::par);
//   
//   // typename ThrustSystem<TBB>::Vector<T> == thrust::host_vector<T>
//   template<typename T>
//   using Vector = thrust::host_vector<T>; // container type
//   
//   template<Depth N>
//   using ConstBitPatternIterator = typename thrust::host_vector<BitPattern<N>>::const_iterator;
//   using ConstDepthIterator = thrust::host_vector<Depth>::const_iterator;
//   using ConstFlagsIterator = thrust::host_vector<Flags>::const_iterator;
//   template<Depth N>
//   using ConstBoxIterator = thrust::zip_iterator<
//                                thrust::tuple<
//                                    ConstBitPatternIterator<N>,
//                                    ConstDepthIterator,
//                                    ConstFlagsIterator>>;
//   
//   template<Depth N>
//   using BitPatternIterator = typename thrust::host_vector<BitPattern<N>>::iterator;
//   using DepthIterator = thrust::host_vector<Depth>::iterator;
//   using FlagsIterator = thrust::host_vector<Flags>::iterator;
//   template<Depth N>
//   using BoxIterator = thrust::zip_iterator<
//                           thrust::tuple<
//                               BitPatternIterator<N>,
//                               DepthIterator,
//                               FlagsIterator>>;
// };

} // namespace b12


#include "ThrustSystem.hpp"
