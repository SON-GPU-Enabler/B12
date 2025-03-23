
#include <iomanip>
#include <limits>
#include <sstream>
#include <type_traits>

#include <thrust/equal.h>
#include <thrust/logical.h>
#include <thrust/sort.h>

#include "BitPattern.h"
#include "Functors.h"


namespace b12 {

template<typename BinaryFunctor>
__host__ __device__
inline BinaryToTupleFunctor<BinaryFunctor>::BinaryToTupleFunctor(BinaryFunctor fun) : fun_(fun) {}

template<typename BinaryFunctor>
__host__ __device__
inline typename BinaryFunctor::result_type BinaryToTupleFunctor<BinaryFunctor>::operator()(
    thrust::tuple<typename BinaryFunctor::first_argument_type, typename BinaryFunctor::second_argument_type> t) const
{
  return fun_(thrust::get<0>(t), thrust::get<1>(t));
}

template<typename BinaryFunctor, typename Iterator1, typename Iterator2>
__host__ __device__
inline thrust::transform_iterator<BinaryToTupleFunctor<BinaryFunctor>,
                                  thrust::zip_iterator<thrust::tuple<Iterator1,Iterator2>>>
makeBinaryTransformIterator(Iterator1 it1, Iterator2 it2, BinaryFunctor fun)
{
  return thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(it1, it2)),
                                         BinaryToTupleFunctor<BinaryFunctor>(fun));
}


template<Architecture A, Depth N, typename BitPatternIterator, typename DepthIterator, typename FlagsIterator>
inline void sortBoxes(BitPatternIterator bitPattern_begin, DepthIterator depth_begin, FlagsIterator flags_begin,
                      NrBoxes n)
{
  if (n >= 1) {
    if (N <= 64) {
      
      bool areAllSameDepth = thrust::all_of(typename ThrustSystem<A>::execution_policy(),
                                            depth_begin, depth_begin + n,
                                            thrust::placeholders::_1 == *depth_begin);
      
      if (areAllSameDepth) {
        
        // radix sort of BitPatterns
        thrust::sort_by_key(typename ThrustSystem<A>::execution_policy(),
                            bitPattern_begin, bitPattern_begin + n,
                            flags_begin);
        
      } else {
        
        bool areAllDepthsSmallerThanMaxDepth = thrust::all_of(typename ThrustSystem<A>::execution_policy(),
                                                              depth_begin, depth_begin + n,
                                                              thrust::placeholders::_1 < getMaxDepth<N>());
        
        auto box_begin = thrust::make_zip_iterator(thrust::make_tuple(bitPattern_begin, depth_begin, flags_begin));
        
        if (areAllDepthsSmallerThanMaxDepth) {
          
          auto it = thrust::make_transform_iterator(box_begin, GetComparableBitPatternFunctor<N>());
          
          typename ThrustSystem<A>::Vector<BitPattern<N>> keys(it, it + n);
          
          // radix sort of boxes
          thrust::sort_by_key(typename ThrustSystem<A>::execution_policy(),
                              keys.begin(), keys.end(),
                              box_begin);
          
        } else {
          
          // merge sort of boxes
          thrust::sort(typename ThrustSystem<A>::execution_policy(),
                       box_begin, box_begin + n,
                       IsStrictlyPrecedingFunctor<N>());
          
        }
      }
      
    } else {
      
      auto box_begin = thrust::make_zip_iterator(thrust::make_tuple(bitPattern_begin, depth_begin, flags_begin));
      
      // merge sort of boxes
      thrust::sort(typename ThrustSystem<A>::execution_policy(),
                   box_begin, box_begin + n,
                   IsStrictlyPrecedingFunctor<N>());
          
    }
  }
}


template<typename T>
inline std::ostream& putNumberIntoJSONStream(std::ostream& os, const std::string& valueName, T value)
{
  if (std::is_floating_point<T>::value) {
    os << std::setprecision(std::numeric_limits<T>::max_digits10);
  }
  
  os << "\n  \"" << valueName << "\": " << value;
  
  return os;
}

template<typename T>
inline std::ostream& putNumberVectorIntoJSONStream(std::ostream& os,
                                                   const std::string& valueName,
                                                   const std::vector<T>& v)
{
  if (std::is_floating_point<T>::value) {
    os << std::setprecision(std::numeric_limits<T>::max_digits10);
  }
  
  os << "\n  \"" << valueName << "\": [";
  
  for (auto it = v.begin(); it != v.end(); ++it) {
    if (it != v.begin()) {
      os << ",";
    }
    os << "\n    " << *it;
  }
  
  os << "\n  ]";
  
  return os;
}

template<typename T>
inline T getNumberFromJSONString(const std::string& str, const std::string& valueName)
{
  T value;
  
  std::string tempString, whitespaces = " \t\n\v\f\r";
  
  std::size_t objectLocation = 0;
  while (objectLocation != std::string::npos && (objectLocation == 0 || str[objectLocation] != ':')) {
    objectLocation = str.find("\"" + valueName + "\"", objectLocation);
    objectLocation = str.find_first_not_of(whitespaces, objectLocation + std::string("\"" + valueName + "\"").length());
  }
  
  if (objectLocation == std::string::npos) {
    std::cout << "Warning: \"" << valueName << "\" could not be found." << std::endl;
  } else {
    whitespaces.append("\"");
    std::size_t start = str.find_first_not_of(whitespaces, objectLocation + 1);
    if (start != std::string::npos) {
      whitespaces.append(",");
      std::size_t end = str.find_first_of(whitespaces, start);
      std::istringstream iss(str.substr(start, end != std::string::npos ? end - start : std::string::npos));
      iss >> value;
      if (iss.fail() || iss.bad()) {
        std::cout << "Warning: Reading the value of \"" << valueName << "\" caused an error." << std::endl;
      }
    } else {
      std::cout << "Warning: Value for \"" << valueName << "\" could not be found." << std::endl;
    }
  }
  
  return value;
}

template<typename T>
inline std::vector<T> getNumberVectorFromJSONString(const std::string& str,
                                                    const std::string& valueName,
                                                    uint64_t n)
{
  std::vector<T> v;
  v.reserve(n);
  
  std::string tempString, whitespaces = " \t\n\v\f\r";
  
  std::size_t objectLocation = 0;
  while (objectLocation != std::string::npos && (objectLocation == 0 || str[objectLocation] != ':')) {
    objectLocation = str.find("\"" + valueName + "\"", objectLocation);
    objectLocation = str.find_first_not_of(whitespaces, objectLocation + std::string("\"" + valueName + "\"").length());
  }
  std::size_t start = str.find('[', objectLocation);
  std::size_t end = str.find(']', objectLocation);
  
  if (objectLocation == std::string::npos || start == std::string::npos) {
    std::cout << "Warning: \"" << valueName << "\" could not be found as an array." << std::endl;
  } else {
    whitespaces.append("\"");
    std::istringstream iss(str.substr(start + 1, end - start - 1));
    std::istringstream issTemp;
    T value;
    
    while (std::getline(iss, tempString, ',')) {
      start = tempString.find_first_not_of(whitespaces);
      if (start != std::string::npos) {
        end = tempString.find_first_of(whitespaces, start);
        issTemp.clear();
        tempString = tempString.substr(start, end != std::string::npos ? end - start : std::string::npos);
        issTemp.str(tempString);
        issTemp >> value;
        v.push_back(value);
        if (issTemp.fail() || issTemp.bad()) {
          std::cout << "Warning: During reading the array \"" << valueName << "\", value " << tempString << " caused an error." << std::endl;
          break;
        }
      }
    }
  }
  
  v.shrink_to_fit();
  
  return v;
}

} // namespace b12
