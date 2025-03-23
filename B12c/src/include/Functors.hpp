
#include "coordinateFunctions.h"


namespace b12 {

__host__ __device__
inline bool AnyIn2TupleFunctor::operator()(thrust::tuple<bool, bool> t) const
{
  return thrust::get<0>(t) || thrust::get<1>(t);
}

__host__ __device__
inline bool AllIn2TupleFunctor::operator()(thrust::tuple<bool, bool> t) const
{
  return thrust::get<0>(t) && thrust::get<1>(t);
}

__host__ __device__
inline bool AllIn3TupleFunctor::operator()(thrust::tuple<bool, bool, bool> t) const
{
  return thrust::get<0>(t) && thrust::get<1>(t) && thrust::get<2>(t);
}

__host__ __device__
inline bool AllIn4TupleFunctor::operator()(thrust::tuple<bool, bool, bool, bool> t) const
{
  return thrust::get<0>(t) && thrust::get<1>(t) && thrust::get<2>(t) && thrust::get<3>(t);
}

__host__ __device__
inline bool IsInvalidNrBoxesFunctor::operator()(NrBoxes n) const
{
  return n == getInvalidNrBoxes();
}

__host__ __device__
inline IDivideByFunctor::IDivideByFunctor(NrPoints divisor) : divisor_(divisor) {}

__host__ __device__
inline NrPoints IDivideByFunctor::operator()(NrPoints n) const
{
  return n / divisor_;
}

__host__ __device__
inline NrBoxes DecrementWhenPositiveFunctor::operator()(NrBoxes n) const
{
  return n - NrBoxes(n > 0);
}

__host__ __device__
inline CapByFunctor::CapByFunctor(NrBoxes upperBound) : upperBound_(upperBound) {}

__host__ __device__
inline NrBoxes CapByFunctor::operator()(NrBoxes n) const
{
  return (n < upperBound_) ?  n : upperBound_;
}

template<typename REAL>
__host__ __device__
inline AdvanceConstRealPointerFunctor<REAL>::AdvanceConstRealPointerFunctor(const REAL * realPointer)
    : realPointer_(realPointer) {}

template<typename REAL>
__host__ __device__
inline const REAL * AdvanceConstRealPointerFunctor<REAL>::operator()(NrPoints n) const
{
  return realPointer_ + n;
}

template<typename REAL>
__host__ __device__
inline AdvanceRealPointerFunctor<REAL>::AdvanceRealPointerFunctor(REAL * realPointer)
    : realPointer_(realPointer) {}

template<typename REAL>
__host__ __device__
inline REAL * AdvanceRealPointerFunctor<REAL>::operator()(NrPoints n) const
{
  return realPointer_ + n;
}

template<typename T1, typename T2>
__host__ __device__
inline T2 PackIntegersFunctor<T1, T2, typename std::enable_if<std::numeric_limits<T2>::digits >=
                                                              2 * std::numeric_limits<T1>::digits>::type>::operator()(
    thrust::tuple<T1, T1> t) const
{
  return (T2(thrust::get<0>(t)) << std::numeric_limits<T1>::digits) | T2(thrust::get<1>(t));
}

template<typename T1, typename T2>
__host__ __device__
inline thrust::tuple<T1, T1>
UnpackIntegersFunctor<T1, T2, typename std::enable_if<std::numeric_limits<T2>::digits >=
                                                      2 * std::numeric_limits<T1>::digits>::type>::operator()(
    T2 n) const
{
  return thrust::make_tuple(T1(n >> std::numeric_limits<T1>::digits), T1(n));
}



template<Depth N>
__host__ __device__
inline BitPattern<N> GetComparableBitPatternFunctor<N>::operator()(Box<N> box) const
{
  return getComparableBitPattern<N>(box);
}

template<Depth N>
__host__ __device__
inline bool HasSamePathFunctor<N>::operator()(Box<N> box1, Box<N> box2) const
{
  return hasSamePath<N>(box1, box2);
}

template<Depth N>
__host__ __device__
inline bool IsInvalidBoxFunctor<N>::operator()(Box<N> box) const
{
  return isInvalidBox<N>(box);
}

template<Depth N>
__host__ __device__
inline bool IsPrecedingInMinorDepthFunctor<N>::operator()(Box<N> box1, Box<N> box2) const
{
  return isPrecedingInMinorDepth<N>(box1, box2);
}

template<Depth N>
__host__ __device__
inline bool IsSameBoxFunctor<N>::operator()(Box<N> box1, Box<N> box2) const
{
  return isSameBox<N>(box1, box2);
}

template<Depth N>
__host__ __device__
inline bool IsStrictlyPrecedingFunctor<N>::operator()(Box<N> box1, Box<N> box2) const
{
  return isStrictlyPreceding<N>(box1, box2);
}



__host__ __device__
inline Flags FlagsAndFunctor::operator()(Flags flags1, Flags flags2) const
{
  return flags1 & flags2;
}

__host__ __device__
inline Flags FlagsOrFunctor::operator()(Flags flags1, Flags flags2) const
{
  return flags1 | flags2;
}

__host__ __device__
inline AreFlagsSetFunctor::AreFlagsSetFunctor(Flags flags) : flags_(flags) {}

__host__ __device__
inline bool AreFlagsSetFunctor::operator()(Flags flags) const
{
  return areFlagsSet(flags, flags_);
}

__host__ __device__
inline IsAnyFlagSetFunctor::IsAnyFlagSetFunctor(Flags flags) : flags_(flags) {}

__host__ __device__
inline bool IsAnyFlagSetFunctor::operator()(Flags flags) const
{
//   return flags.isAnyFlagSet(flags_);
  return isAnyFlagSet(flags, flags_);
}

__host__ __device__
inline ChangeFlagsFunctor::ChangeFlagsFunctor(Flags flagsToUnset, Flags flagsToSet)
    : flagsToUnset_(flagsToUnset), flagsToSet_(flagsToSet) {}

__host__ __device__
inline Flags ChangeFlagsFunctor::operator()(Flags flags) const
{
  return change(flags, flagsToUnset_, flagsToSet_); // if f is set, unset f and set g
}

__host__ __device__
inline SetFlagsFunctor::SetFlagsFunctor(Flags flags) : flags_(flags) {}

__host__ __device__
inline Flags SetFlagsFunctor::operator()(Flags flags) const
{
  return set(flags, flags_);
}

__host__ __device__
inline UnsetFlagsFunctor::UnsetFlagsFunctor(Flags flags) : flags_(flags) {}

__host__ __device__
inline Flags UnsetFlagsFunctor::operator()(Flags flags) const
{
  return unset(flags, flags_);
}



template<Depth N>
__host__ __device__
inline IncrementDepthByFunctor<N>::IncrementDepthByFunctor(Depth depth) : depth_(depth) {}

template<Depth N>
__host__ __device__
inline Box<N> IncrementDepthByFunctor<N>::operator()(Box<N> box) const
{
  return incrementDepthBy<N>(box, depth_);
}

template<Depth N>
__host__ __device__
inline DecrementDepthByFunctor<N>::DecrementDepthByFunctor(Depth depth) : depth_(depth) {}

template<Depth N>
__host__ __device__
inline Box<N> DecrementDepthByFunctor<N>::operator()(Box<N> box) const
{
  return decrementDepthBy<N>(box, depth_);
}

template<Depth N>
__host__ __device__
inline DecrementDepthToFunctor<N>::DecrementDepthToFunctor(Depth depth) : depth_(depth) {}

template<Depth N>
__host__ __device__
inline Box<N> DecrementDepthToFunctor<N>::operator()(Box<N> box) const
{
  return decrementDepthTo<N>(box, depth_);
}

template<Depth N>
__host__ __device__
inline ChangeDepthToFunctor<N>::ChangeDepthToFunctor(Depth depth) : depth_(depth) {}

template<Depth N>
__host__ __device__
inline Box<N> ChangeDepthToFunctor<N>::operator()(Box<N> box) const
{
  return changeDepthTo<N>(box, depth_);
}

template<Depth N>
__host__ __device__
inline HasAtLeastDepthFunctor<N>::HasAtLeastDepthFunctor(Depth depth) : depth_(depth) {}

template<Depth N>
__host__ __device__
inline bool HasAtLeastDepthFunctor<N>::operator()(Box<N> box) const
{
  return hasAtLeastDepth<N>(box, depth_);
}

template<Depth N>
__host__ __device__
inline Depth ComputeDepthOfMutualAncestorFunctor<N>::operator()(Box<N> box1, Box<N> box2) const
{
  return computeDepthOfMutualAncestor<N>(box1, box2);
}

template<Depth N>
__host__ __device__
inline IsUnequalToPredecessorInDepthFunctor<N>::IsUnequalToPredecessorInDepthFunctor(Depth depth)
    : depth_(depth) {}

template<Depth N>
__host__ __device__
inline bool IsUnequalToPredecessorInDepthFunctor<N>::operator()(thrust::tuple<NrBoxes, Box<N>, Box<N>> t) const
{
  return hasAtLeastDepth<N>(thrust::get<1>(t), depth_)
         &&
         (thrust::get<0>(t) == NrBoxes(0) || !isSameBox<N>(thrust::get<1>(t), thrust::get<2>(t)));
}

template<Depth N>
__host__ __device__
inline NrBoxes ComputeSearchIndexFunctor<N>::operator()(
    thrust::tuple<NrBoxes, Box<N>, Box<N>, Box<N>, bool, bool> t) const
{
  NrBoxes res;
  
  if (thrust::get<4>(t) &&
      getDepth<N>(thrust::get<1>(t)) >= getDepth<N>(thrust::get<2>(t)) && 
      hasSamePath<N>(thrust::get<1>(t), thrust::get<2>(t))) {
    
    // first Box is "similar" to second box
    res = thrust::get<0>(t) > NrBoxes(0) ? thrust::get<0>(t) - NrBoxes(1) : NrBoxes(0);
    
  } else if (thrust::get<5>(t) &&
             getDepth<N>(thrust::get<1>(t)) >= getDepth<N>(thrust::get<3>(t)) && 
             hasSamePath<N>(thrust::get<1>(t), thrust::get<3>(t))) {
    
    // first Box is "similar" to third box
    res = thrust::get<0>(t);
    
  } else {
    
    res = getInvalidNrBoxes();
    
  }
  
  return res;
}



template<Depth N>
__host__ __device__
inline Box<N> ComputeOtherHalfFunctor<N>::operator()(Box<N> box) const
{
  return computeOtherHalf<N>(box);
}

template<Depth N>
__host__ __device__
inline IsSubdividableFunctor<N>::IsSubdividableFunctor(Flags flags) : flags_(flags) {}

template<Depth N>
__host__ __device__
inline bool IsSubdividableFunctor<N>::operator()(Box<N> box) const
{
  return areFlagsSet(getFlags<N>(box), flags_) && getDepth<N>(box) < getMaxDepth<N>();
}

template<Depth N>
__host__ __device__
inline IsUnsubdividableFunctor<N>::IsUnsubdividableFunctor(Flags flags) : flags_(flags) {}

template<Depth N>
__host__ __device__
inline bool IsUnsubdividableFunctor<N>::operator()(thrust::tuple<Box<N>, Box<N>, Box<N>> t) const
{
  // t = tuple<Box, pred_box, succ_box>
  
  bool res = areFlagsSet(getFlags<N>(thrust::get<0>(t)), flags_) && getDepth<N>(thrust::get<0>(t)) > Depth(0);
  
  if (res) {
    // if the box and its neigbour are siblings, check for the Flags, 
    // else, check if the parent of the box and the neighbour are not on the same path
    if (isRightHalf<N>(thrust::get<0>(t))) {
      // take second box in tuple t
      res = isSameBox<N>(decrementDepthBy<N>(thrust::get<0>(t), 1), decrementDepthBy<N>(thrust::get<1>(t), 1))
            ? areFlagsSet(getFlags<N>(thrust::get<1>(t)), flags_)
            : ! hasSamePath<N>(decrementDepthBy<N>(thrust::get<0>(t), 1), thrust::get<1>(t));
    } else {
      // take third box in tuple t
      res = isSameBox<N>(decrementDepthBy<N>(thrust::get<0>(t), 1), decrementDepthBy<N>(thrust::get<2>(t), 1))
            ? areFlagsSet(getFlags<N>(thrust::get<2>(t)), flags_)
            : ! hasSamePath<N>(decrementDepthBy<N>(thrust::get<0>(t), 1), thrust::get<2>(t));
    }
  }
  
  return res;
  
//   // the version below causes use of local memory, most probably due to the reference in the first line
//   Box<N>& neighbour = isRightHalf<N>(thrust::get<0>(t)) ? thrust::get<1>(t) : thrust::get<2>(t);
//   
//   return areFlagsSet(getFlags<N>(thrust::get<0>(t)), flags_)
//          && 
//          getDepth<N>(thrust::get<0>(t)) > Depth(0)
//          &&
//          (isSameBox<N>(decrementDepthBy<N>(thrust::get<0>(t), 1), decrementDepthBy<N>(neighbour, 1))
//           ? areFlagsSet(getFlags<N>(neighbour), flags_)
//           : ! hasSamePath<N>(decrementDepthBy<N>(thrust::get<0>(t), 1), neighbour));
}

template<Depth N>
__host__ __device__
inline Box<N> AdaptDepthOfFirstBoxFunctor<N>::operator()(thrust::tuple<Box<N>, Box<N>, Box<N>> t) const
{
  // t = tuple<new Box, Box[search_index-1], Box[search_index]>
  return changeDepthTo<N>(thrust::get<0>(t),
                          thrust::maximum<Depth>()(
                              computeDepthOfMutualAncestor<N>(thrust::get<0>(t), thrust::get<1>(t)),
                              computeDepthOfMutualAncestor<N>(thrust::get<0>(t), thrust::get<2>(t))
                          ) + Depth(1));
}

// computes Box(max(getBitPattern(box1),getBitPattern(box2)),
//              max(getDepth(box1),getDepth(box2)),
//              or(getFlags(box1),getFlags(box2)))
template<Depth N>
__host__ __device__
inline Box<N> ComputeMaxBitPatternMaxDepthOrFlagsFunctor<N>::operator()(Box<N> box1, Box<N> box2) const
{
  return Box<N>(thrust::maximum<BitPattern<N>>()(getBitPattern<N>(box1), getBitPattern<N>(box2)),
                thrust::maximum<Depth>()(getDepth<N>(box1), getDepth<N>(box2)),
                getFlags<N>(box1) | getFlags<N>(box2));
}



template<Dimension DIM, typename REAL>
__host__ __device__
inline IsPointInBoxFunctor<DIM, REAL>::IsPointInBoxFunctor(const REAL * center, const REAL * radius)
    : center_(center), radius_(radius) {}

template<Dimension DIM, typename REAL>
__host__ __device__
inline bool IsPointInBoxFunctor<DIM, REAL>::operator()(const REAL * point) const
{
  return isPointInBox<DIM, REAL>(point, center_, radius_);
}


template<Dimension DIM, typename REAL, Depth N>
__host__ __device__
inline ComputeBoxFromPointFunctor<DIM, REAL, N>::ComputeBoxFromPointFunctor(const REAL * center, 
                                                                            const REAL * radius, 
                                                                            const Dimension * sdScheme,
                                                                            const Depth * sdCount,
                                                                            bool isSdSchemeDefault,
                                                                            Flags flags)
    : center_(center), radius_(radius), sdScheme_(sdScheme), sdCount_(sdCount),
      isSdSchemeDefault_(isSdSchemeDefault), flags_(flags) {}

template<Dimension DIM, typename REAL, Depth N>
__host__ __device__
inline Box<N> ComputeBoxFromPointFunctor<DIM, REAL, N>::operator()(const REAL * point) const
{
  return isSdSchemeDefault_ ? computeBoxFromPointFixedSdScheme<DIM, REAL, N>(point, center_, radius_, flags_)
                            : computeBoxFromPoint<DIM, REAL, N>(point,
                                                                center_, radius_, sdScheme_, sdCount_,
                                                                flags_);
}

template<Dimension DIM, typename REAL, Depth N>
__host__ __device__
inline ComputeCoordinatesOfBoxFunctor<DIM, REAL, N>::ComputeCoordinatesOfBoxFunctor(const REAL * center, 
                                                                                    const REAL * radius, 
                                                                                    const Dimension * sdScheme,
                                                                                    const Depth * sdCount,
                                                                                    bool isSdSchemeDefault)
    : center_(center), radius_(radius), sdScheme_(sdScheme), sdCount_(sdCount),
      isSdSchemeDefault_(isSdSchemeDefault) {}

template<Dimension DIM, typename REAL, Depth N>
__host__ __device__
inline void ComputeCoordinatesOfBoxFunctor<DIM, REAL, N>::operator()(thrust::tuple<Box<N>, REAL *> t) const
{
  if (isSdSchemeDefault_) {
    computeCoordinatesOfBoxFixedSdScheme<DIM, REAL, N>(thrust::get<0>(t), center_, radius_, thrust::get<1>(t));
  } else {
    computeCoordinatesOfBox<DIM, REAL, N>(thrust::get<0>(t),
                                          center_, radius_, sdScheme_, sdCount_,
                                          thrust::get<1>(t));
  }
}

template<Dimension DIM, typename REAL, Depth N, typename Grid, typename Map>
__host__ __device__
inline GridPointsFunctor<DIM, REAL, N, Grid, Map>::GridPointsFunctor(Grid grid,
                                                                     Map map,
                                                                     Flags flags,
                                                                     const REAL * center,
                                                                     const REAL * radius,
                                                                     const Dimension * sdScheme,
                                                                     const Depth * sdCount,
                                                                     bool isSdSchemeDefault)
    : grid_(grid), map_(map), flags_(flags),
      center_(center) , radius_(radius),
      sdScheme_(sdScheme), sdCount_(sdCount), isSdSchemeDefault_(isSdSchemeDefault) {}

template<Dimension DIM, typename REAL, Depth N, typename Grid, typename Map>
__host__ __device__
inline Box<N> GridPointsFunctor<DIM, REAL, N, Grid, Map>::operator()(thrust::tuple<NrPoints, Box<N>> t) const
{
  // coords[0     ... DIM-1]   == center of box;
  // coords[DIM   ... 2*DIM-1] == radius of box;
  // coords[2*DIM ... 3*DIM-1] == coordinates of sample grid point;
  REAL coords[3 * DIM];
  
  if (isSdSchemeDefault_) {
    computeCoordinatesOfBoxFixedSdScheme<DIM, REAL, N>(thrust::get<1>(t), center_, radius_, coords);
  } else {
    computeCoordinatesOfBox<DIM, REAL, N>(thrust::get<1>(t),
                                          center_, radius_, sdScheme_, sdCount_,
                                          coords);
  }
  
  // compute coordinates of sample grid point
  grid_(thrust::get<0>(t), coords + 2 * DIM);
  
#pragma unroll
  for (Dimension i = 0; i < DIM; ++i) {
    // res[i] = grid[i] * radius[i] + center[i];
    // res gets stored in the "center"-region of coords, 
    // i.e. coords[i] = coords[i + 2*DIM] * coords[i + DIM] + coords[i];
    coords[i] += coords[i + 2 * DIM] * coords[i + DIM];
  }
  
  // map grid point and store it in coords[DIM ... 2*DIM-1]
  map_(coords, coords + DIM);
  
  return isSdSchemeDefault_ ? computeBoxFromPointFixedSdScheme<DIM, REAL, N>(coords + DIM, center_, radius_, flags_)
                            : computeBoxFromPoint<DIM, REAL, N>(coords + DIM,
                                                                center_, radius_, sdScheme_, sdCount_,
                                                                flags_);
}

} // namespace b12
