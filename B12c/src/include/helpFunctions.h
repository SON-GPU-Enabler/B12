
#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include "ThrustSystem.h"
#include "TypeDefinitions.h"


namespace b12 {

// "conversion" functor that takes a 2-tuple and applies the given BinaryFunctor to its elements
template<typename BinaryFunctor>
struct BinaryToTupleFunctor : public thrust::unary_function<thrust::tuple<typename BinaryFunctor::first_argument_type,
                                                                          typename BinaryFunctor::second_argument_type>,
                                                            typename BinaryFunctor::result_type>
{
  BinaryFunctor fun_;
  
  __host__ __device__
  BinaryToTupleFunctor(BinaryFunctor fun);
  
  __host__ __device__
  typename BinaryFunctor::result_type operator()(
      thrust::tuple<typename BinaryFunctor::first_argument_type,
                    typename BinaryFunctor::second_argument_type> t) const;
};

// creates an iterator that represents a pointer into a range of values after transformation by a binary function
template<typename BinaryFunctor, typename Iterator1, typename Iterator2>
__host__ __device__
thrust::transform_iterator<BinaryToTupleFunctor<BinaryFunctor>,
                           thrust::zip_iterator<thrust::tuple<Iterator1,Iterator2>>>
makeBinaryTransformIterator(Iterator1 it1, Iterator2 it2, BinaryFunctor fun);


// sort boxes in the range [_begin, _begin + n)
template<Architecture A, Depth N, typename BitPatternIterator, typename DepthIterator, typename FlagsIterator>
void sortBoxes(BitPatternIterator bitPattern_begin, DepthIterator depth_begin, FlagsIterator flags_begin, NrBoxes n);


// writes a value into an outstream in JSON style
template<typename T>
std::ostream& putNumberIntoJSONStream(std::ostream& os, const std::string& valueName, T value);

// writes a std::vector into an outstream in JSON style
template<typename T>
std::ostream& putNumberVectorIntoJSONStream(std::ostream& os, const std::string& valueName, const std::vector<T>& v);

// reads a value from a JSON style string
template<typename T>
T getNumberFromJSONString(const std::string& str, const std::string& valueName);

// reads a std::vector from a JSON style string; before filling, n elements are reserved in the resulting vector
template<typename T>
std::vector<T> getNumberVectorFromJSONString(const std::string& str, const std::string& valueName, uint64_t n = 0);

} // namespace b12


#include "helpFunctions.hpp"
