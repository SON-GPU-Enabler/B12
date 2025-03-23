
#pragma once

#include <limits>
#include <type_traits>

#include <thrust/functional.h>
#include <thrust/tuple.h>

#include "BitPattern.h"
#include "Box.h"
#include "Flags.h"
#include "TypeDefinitions.h"


namespace b12 {

// Functors mainly used for thrust algorithms; therefore no pass-by-reference

  
/// helper functors ---------------------------------------------------------------------------------------------------

// returns true iff one of both elements of tuple<bool, bool> are true
struct AnyIn2TupleFunctor : public thrust::unary_function<thrust::tuple<bool, bool>, bool>
{
  __host__ __device__
  bool operator()(thrust::tuple<bool, bool> t) const;
};

// returns true iff both elements of tuple<bool, bool> are true
struct AllIn2TupleFunctor : public thrust::unary_function<thrust::tuple<bool, bool>, bool>
{
  __host__ __device__
  bool operator()(thrust::tuple<bool, bool> t) const;
};

// returns true iff all elements of tuple<bool, bool, bool, bool> are true
struct AllIn3TupleFunctor : public thrust::unary_function<thrust::tuple<bool, bool, bool>, bool>
{
  __host__ __device__
  bool operator()(thrust::tuple<bool, bool, bool> t) const;
};

// returns true iff all elements of tuple<bool, bool, bool, bool> are true
struct AllIn4TupleFunctor : public thrust::unary_function<thrust::tuple<bool, bool, bool, bool>, bool>
{
  __host__ __device__
  bool operator()(thrust::tuple<bool, bool, bool, bool> t) const;
};

// checks whether the input is equal to getInvalidNrBoxes()
struct IsInvalidNrBoxesFunctor : public thrust::unary_function<NrBoxes, bool>
{
  __host__ __device__
  bool operator()(NrBoxes n) const;
};

// returns input/divisor_
struct IDivideByFunctor : public thrust::unary_function<NrPoints, NrPoints>
{
  NrPoints divisor_;
  
  __host__ __device__
  IDivideByFunctor(NrPoints divisor);
  
  __host__ __device__
  NrPoints operator()(NrPoints n) const;
};

// returns max(input - 1, 0)
struct DecrementWhenPositiveFunctor : public thrust::unary_function<NrBoxes, NrBoxes>
{
  __host__ __device__
  NrBoxes operator()(NrBoxes n) const;
};

// returns min(input, upper_bound)
struct CapByFunctor : public thrust::unary_function<NrBoxes, NrBoxes>
{
  NrBoxes upperBound_;
  
  __host__ __device__
  CapByFunctor(NrBoxes upperBound);
  
  __host__ __device__
  NrBoxes operator()(NrBoxes n) const;
};

// returns real_ptr + n
template<typename REAL>
struct AdvanceConstRealPointerFunctor : public thrust::unary_function<NrPoints, const REAL *>
{
  const REAL * realPointer_;
  
  __host__ __device__
  AdvanceConstRealPointerFunctor(const REAL * realPointer);
  
  __host__ __device__
  const REAL * operator()(NrPoints n) const;
};

// returns real_ptr + n
template<typename REAL>
struct AdvanceRealPointerFunctor : public thrust::unary_function<NrPoints, REAL *>
{
  REAL * realPointer_;
  
  __host__ __device__
  AdvanceRealPointerFunctor(REAL * realPointer);
  
  __host__ __device__
  REAL * operator()(NrPoints n) const;
};

template<typename T1, typename T2, typename T3 = void>
struct PackIntegersFunctor
{
  using IsValid = std::false_type;
  
  __host__ __device__
  T2 operator()(thrust::tuple<T1, T1> t) const;
};

// wraps a tuple of two integers of type T1 into a integer of type T2
template<typename T1, typename T2>
struct PackIntegersFunctor<T1, T2, typename std::enable_if<std::numeric_limits<T2>::digits >=
                                                           2 * std::numeric_limits<T1>::digits>::type>
    : public thrust::unary_function<thrust::tuple<T1, T1>, T2>
{
  using IsValid = std::true_type;
  
  __host__ __device__
  T2 operator()(thrust::tuple<T1, T1> t) const;
};

template<typename T1, typename T2, typename T3 = void>
struct UnpackIntegersFunctor
{
  using IsValid = std::false_type;
  
  __host__ __device__
  thrust::tuple<T1, T1> operator()(T2 n) const;
};;

// inverse method of PackIntegersFunctor
template<typename T1, typename T2>
struct UnpackIntegersFunctor<T1, T2, typename std::enable_if<std::numeric_limits<T2>::digits >=
                                                             2 * std::numeric_limits<T1>::digits>::type>
    : public thrust::unary_function<T2, thrust::tuple<T1, T1>>
{
  using IsValid = std::true_type;
  
  __host__ __device__
  thrust::tuple<T1, T1> operator()(T2 n) const;
};



/// comparison functors -----------------------------------------------------------------------------------------------

template<Depth N>
struct GetComparableBitPatternFunctor : public thrust::unary_function<Box<N>, BitPattern<N>>
{
  __host__ __device__
  BitPattern<N> operator()(Box<N> box) const;
};

template<Depth N>
struct HasSamePathFunctor : public thrust::binary_function<Box<N>, Box<N>, bool>
{
  __host__ __device__
  bool operator()(Box<N> box1, Box<N> box2) const;
};

template<Depth N>
struct IsInvalidBoxFunctor : public thrust::unary_function<Box<N>, bool>
{
  __host__ __device__
  bool operator()(Box<N> box) const;
};

template<Depth N>
struct IsPrecedingInMinorDepthFunctor : public thrust::binary_function<Box<N>, Box<N>, bool>
{
  __host__ __device__
  bool operator()(Box<N> box1, Box<N> box2) const;
};

template<Depth N>
struct IsSameBoxFunctor : public thrust::binary_function<Box<N>, Box<N>, bool>
{
  __host__ __device__
  bool operator()(Box<N> box1, Box<N> box2) const;
};

template<Depth N>
struct IsStrictlyPrecedingFunctor : public thrust::binary_function<Box<N>, Box<N>, bool>
{
  __host__ __device__
  bool operator()(Box<N> box1, Box<N> box2) const;
};



/// Flags operations --------------------------------------------------------------------------------------------------

struct FlagsAndFunctor : public thrust::binary_function<Flags, Flags, Flags>
{
  __host__ __device__
  Flags operator()(Flags flags1, Flags flags2) const;
};

struct FlagsOrFunctor : public thrust::binary_function<Flags, Flags, Flags>
{
  __host__ __device__
  Flags operator()(Flags flags1, Flags flags2) const;
};

struct AreFlagsSetFunctor : public thrust::unary_function<Flags, bool>
{
  Flags flags_;
  
  __host__ __device__
  AreFlagsSetFunctor(Flags flags);
  
  __host__ __device__
  bool operator()(Flags flags) const;
};

struct IsAnyFlagSetFunctor : public thrust::unary_function<Flags, bool>
{
  Flags flags_;
  
  __host__ __device__
  IsAnyFlagSetFunctor(Flags flags);
  
  __host__ __device__
  bool operator()(Flags flags) const;
};

struct ChangeFlagsFunctor : public thrust::unary_function<Flags, Flags>
{
  Flags flagsToUnset_, flagsToSet_;
  
  __host__ __device__
  ChangeFlagsFunctor(Flags flagsToUnset, Flags flagsToSet);
  
  __host__ __device__
  Flags operator()(Flags flags) const;
};

struct SetFlagsFunctor : public thrust::unary_function<Flags, Flags>
{
  Flags flags_;
  
  __host__ __device__
  SetFlagsFunctor(Flags flags);
  
  __host__ __device__
  Flags operator()(Flags flags) const;
};

struct UnsetFlagsFunctor : public thrust::unary_function<Flags, Flags>
{
  Flags flags_;
  
  __host__ __device__
  UnsetFlagsFunctor(Flags flags);
  
  __host__ __device__
  Flags operator()(Flags flags) const;
};


/// Depth operations --------------------------------------------------------------------------------------------------

template<Depth N>
struct IncrementDepthByFunctor : public thrust::unary_function<Box<N>, Box<N>>
{
  Depth depth_;
  
  __host__ __device__
  IncrementDepthByFunctor(Depth depth);
  
  __host__ __device__
  Box<N> operator()(Box<N> box) const;
};

template<Depth N>
struct DecrementDepthByFunctor : public thrust::unary_function<Box<N>, Box<N>>
{
  Depth depth_;
  
  __host__ __device__
  DecrementDepthByFunctor(Depth depth);
  
  __host__ __device__
  Box<N> operator()(Box<N> box) const;
};

template<Depth N>
struct DecrementDepthToFunctor : public thrust::unary_function<Box<N>, Box<N>>
{
  Depth depth_;
  
  __host__ __device__
  DecrementDepthToFunctor(Depth depth);
  
  __host__ __device__
  Box<N> operator()(Box<N> box) const;
};

template<Depth N>
struct ChangeDepthToFunctor : public thrust::unary_function<Box<N>, Box<N>>
{
  Depth depth_;
  
  __host__ __device__
  ChangeDepthToFunctor(Depth depth);
  
  __host__ __device__
  Box<N> operator()(Box<N> box) const;
};

template<Depth N>
struct HasAtLeastDepthFunctor : public thrust::unary_function<Box<N>, bool>
{
  Depth depth_;
  
  __host__ __device__
  HasAtLeastDepthFunctor(Depth depth);
  
  __host__ __device__
  bool operator()(Box<N> box) const;
};

template<Depth N>
struct ComputeDepthOfMutualAncestorFunctor : public thrust::binary_function<Box<N>, Box<N>, Depth>
{
  __host__ __device__
  Depth operator()(Box<N> box1, Box<N> box2) const;
};

// checks whether a Box (thrust::get<1>(t)) is in depth d different to its preceding Box (thrust::get<2>(t))
// if Box has smaller depth than d, 0 is the result
// Note: thrust::get<0>(t) indicates the index of thrust::get<1>(t)
template<Depth N>
struct IsUnequalToPredecessorInDepthFunctor : public thrust::unary_function<thrust::tuple<NrBoxes, Box<N>, Box<N>>,
                                                                            bool>
{
  Depth depth_;
  
  __host__ __device__
  IsUnequalToPredecessorInDepthFunctor(Depth depth);
  
  __host__ __device__
  bool operator()(thrust::tuple<NrBoxes, Box<N>, Box<N>> t) const;
};

// returns the "true" search index out of lower_bound index, i.e.
// if the first box is part of the second box and the first bool is true, max(first arg. - 1, 0) is returned,
// otherwise if the first box is part of the third box and the second bool is true, the first arg. is returned,
// otherwise Invalid Box is returned
template<Depth N>
struct ComputeSearchIndexFunctor : public thrust::unary_function<thrust::tuple<NrBoxes,
                                                                               Box<N>, Box<N>, Box<N>,
                                                                               bool, bool>,
                                                                 NrBoxes>
{
  __host__ __device__
  NrBoxes operator()(thrust::tuple<NrBoxes, Box<N>, Box<N>, Box<N>, bool, bool> t) const;
};


/// "tree" operations -------------------------------------------------------------------------------------------------

// Resulting box is the other half of the box containing the input box as one half
// Surrounding Box and Invalid Box have no other half
template<Depth N>
struct ComputeOtherHalfFunctor : public thrust::unary_function<Box<N>, Box<N>>
{
  __host__ __device__
  Box<N> operator()(Box<N> box) const;
};

// checks whether a Box is subdividable, i.e. if flags f are set and depth < getMaxDepth()
template<Depth N>
struct IsSubdividableFunctor : public thrust::unary_function<Box<N>, bool>
{
  Flags flags_;
  
  __host__ __device__
  IsSubdividableFunctor(Flags flags);
  
  __host__ __device__
  bool operator()(Box<N> box) const;
};

// checks whether a Box is unsubdividable
// Input: tuple< Box to analyse, Box' predecessor, Box' successor >
// Note: first box' predecessor must be equal to the first box; same for the last box and its successor
template<Depth N>
struct IsUnsubdividableFunctor : public thrust::unary_function<thrust::tuple<Box<N>, Box<N>, Box<N>>, bool>
{
  Flags flags_;
  
  __host__ __device__
  IsUnsubdividableFunctor(Flags flags);
  
  __host__ __device__
  bool operator()(thrust::tuple<Box<N>, Box<N>, Box<N>> t) const;
};

// changes the depth of a newly inserted leaf so that it suits best in the tree 
// Input: tuple< new Box, Box[search_index-1], Box[search_index] >
// Note: if search_index==0, then Box[search_index-1] should be Box[0]
// Note: if search_index==end, then Box[search_index] should be Box[end-1]
template<Depth N>
struct AdaptDepthOfFirstBoxFunctor : public thrust::unary_function<thrust::tuple<Box<N>, Box<N>, Box<N>>, Box<N>>
{
  __host__ __device__
  Box<N> operator()(thrust::tuple<Box<N>, Box<N>, Box<N>> t) const;
};

// computes Box(max(getBitPattern(box1),getBitPattern(box2)),
//              max(getDepth(box1),getDepth(box2)),
//              or(getFlags(box1),getFlags(box2)))
template<Depth N>
struct ComputeMaxBitPatternMaxDepthOrFlagsFunctor : public thrust::binary_function<Box<N>, Box<N>, Box<N>>
{
  __host__ __device__
  Box<N> operator()(Box<N> box1, Box<N> box2) const;
};


/// functors for: coordinates <--> box --------------------------------------------------------------------------------

// checks whether a point lies a box
template<Dimension DIM, typename REAL>
struct IsPointInBoxFunctor : public thrust::unary_function<REAL *, bool>
{
  const REAL * center_;
  const REAL * radius_;
  
  __host__ __device__
  IsPointInBoxFunctor(const REAL * center, const REAL * radius);
  
  __host__ __device__
  bool operator()(const REAL * point) const;
};

// computes the (possibly imaginary) box which a given point lies in
template<Dimension DIM, typename REAL, Depth N>
struct ComputeBoxFromPointFunctor : public thrust::unary_function<REAL *, Box<N>>
{
  const REAL * center_;
  const REAL * radius_;
  const Dimension * sdScheme_;
  const Depth * sdCount_;
  bool isSdSchemeDefault_;
  Flags flags_;
  
  __host__ __device__
  ComputeBoxFromPointFunctor(const REAL * center, 
                             const REAL * radius, 
                             const Dimension * sdScheme,
                             const Depth * sdCount,
                             bool isSdSchemeDefault,
                             Flags flags = Flag::NONE);
  
  __host__ __device__
  Box<N> operator()(const REAL * point) const;
};

// computes the center, radius (and Flags) of a box
template<Dimension DIM, typename REAL, Depth N>
struct ComputeCoordinatesOfBoxFunctor
{
  const REAL * center_;
  const REAL * radius_;
  const Dimension * sdScheme_;
  const Depth * sdCount_;
  bool isSdSchemeDefault_;
  
  __host__ __device__
  ComputeCoordinatesOfBoxFunctor(const REAL * center, 
                                 const REAL * radius, 
                                 const Dimension * sdScheme,
                                 const Depth * sdCount,
                                 bool isSdSchemDefault);
  
  __host__ __device__
  void operator()(thrust::tuple<Box<N>, REAL *> t) const;
};

// applied to thrust::counting_iterator<NrPoints>(0) the resulting transform_iterator is an iterator for 
// all grid points in the ImplicitBoxTree that have Flags flagsToCheck_ set
template<Dimension DIM, typename REAL, Depth N, typename Grid, typename Map>
struct GridPointsFunctor : public thrust::unary_function<thrust::tuple<NrPoints, Box<N>>, Box<N>>
{
  Grid grid_;
  Map map_;
  Flags flags_; // Flags that mapped boxes get
  const REAL * center_; // center of surrounding box
  const REAL * radius_; // radius of surrounding box
  const Dimension * sdScheme_;
  const Depth * sdCount_;
  bool isSdSchemeDefault_;
  
  __host__ __device__
  GridPointsFunctor(Grid grid,
                    Map map,
                    Flags flags,
                    const REAL * center, 
                    const REAL * radius, 
                    const Dimension * sdScheme,
                    const Depth * sdCount,
                    bool isSdSchemDefault);
  
  __host__ __device__
  Box<N> operator()(thrust::tuple<NrPoints, Box<N>> t) const;
};

} // namespace b12


#include "Functors.hpp"
