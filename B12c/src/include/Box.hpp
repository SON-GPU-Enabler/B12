
#include <thrust/functional.h>


namespace b12 {

template<Depth N>
__host__ __device__
inline Box<N> getSurroundingBox()
{
  return Box<N>(BitPattern<N>(0), Depth(0), Flag::NONE);
}

template<Depth N>
__host__ __device__
inline Box<N> getInvalidBox()
{
  return Box<N>(BitPattern<N>(-1), getLeafDepth(), Flag::NONE);
}


template<Depth N>
__host__ __device__
inline const BitPattern<N>& getBitPattern(const Box<N>& box)
{
  return thrust::get<0>(box);
}

template<Depth N>
__host__ __device__
inline BitPattern<N>& setBitPattern(Box<N>& box)
{
  return thrust::get<0>(box);
}

template<Depth N>
__host__ __device__
inline const Depth& getDepth(const Box<N>& box)
{
  return thrust::get<1>(box);
}

template<Depth N>
__host__ __device__
inline Depth& setDepth(Box<N>& box)
{
  return thrust::get<1>(box);
}

template<Depth N>
__host__ __device__
inline const Flags& getFlags(const Box<N>& box)
{
  return thrust::get<2>(box);
}

template<Depth N>
__host__ __device__
inline Flags& setFlags(Box<N>& box)
{
  return thrust::get<2>(box);
}


template<Depth N>
__host__ __device__
inline BitPattern<N> getComparableBitPattern(const Box<N>& box)
{
  int d = int(getMaxDepth<N>()) - int(getDepth<N>(box));
  
  return safeLeftShift<N>(getBitPattern<N>(box), d) | (safeLeftShift<N>(BitPattern<N>(1), d - 1) - BitPattern<N>(1));
}

template<Depth N>
__host__ __device__
inline bool hasSamePath(const Box<N>& box1, const Box<N>& box2)
{
  // box1.BitPattern >> max(box1.Depth-box2.Depth,0) == box2.BitPattern >> max(box2.Depth-box1.Depth,0);
  int diff = int(getDepth<N>(box1)) - int(getDepth<N>(box2));
  return isInvalidBox<N>(box1) && isInvalidBox<N>(box2)
         ||
         safeRightShift<N>(getBitPattern<N>(box1), diff) == safeRightShift<N>(getBitPattern<N>(box2), -diff);
}

template<Depth N>
__host__ __device__
inline bool isInvalidBox(const Box<N>& box)
{
  return getDepth<N>(box) > getMaxDepth<N>();
}

template<Depth N>
__host__ __device__
inline bool isSurroundingBox(const Box<N>& box)
{
  return getDepth<N>(box) == Depth(0) && getBitPattern<N>(box) == BitPattern<N>(0);
}

template<Depth N>
__host__ __device__
inline bool isPrecedingInMinorDepth(const Box<N>& box1, const Box<N>& box2)
{
  // box1.BitPattern>>max(box1.Depth-box2.Depth,0) < box2.BitPattern>>max(box2.Depth-box1.Depth,0);
  int diff = int(getDepth<N>(box1)) - int(getDepth<N>(box2));
  return !isInvalidBox<N>(box1) // if box1 is invalid then false is always returned
         &&
         (isInvalidBox<N>(box2) // box1 is valid -> return true if box2 is invalid
          ||
          // both boxes valid -> compare
          safeRightShift<N>(getBitPattern<N>(box1), diff) < safeRightShift<N>(getBitPattern<N>(box2), -diff));
}

template<Depth N>
__host__ __device__
inline bool isSameBox(const Box<N>& box1, const Box<N>& box2)
{
  return isInvalidBox<N>(box1) && isInvalidBox<N>(box2)
         ||
         getBitPattern<N>(box1) == getBitPattern<N>(box2) && getDepth<N>(box1) == getDepth<N>(box2);
}

template<Depth N>
__host__ __device__
inline bool isStrictlyPreceding(const Box<N>& box1, const Box<N>& box2)
{
  BitPattern<N> box1Bits = getComparableBitPattern<N>(box1);
  BitPattern<N> box2Bits = getComparableBitPattern<N>(box2);
  
  return !isInvalidBox<N>(box1) // if box1 is invalid then false is always returned
         &&
         (isInvalidBox<N>(box2) // box1 is valid -> return true if box2 is invalid
          ||
          (box1Bits < box2Bits || 
           box1Bits == box2Bits && getDepth<N>(box2) < getDepth<N>(box1))); // boxes valid => compare
}


template<Depth N>
__host__ __device__
inline Box<N> incrementDepthBy(const Box<N>& box, Depth depth)
{
  depth = thrust::minimum<Depth>()(depth, getMaxDepth<N>() - getDepth<N>(box));
  
  return Box<N>(safeLeftShift<N>(getBitPattern<N>(box), depth), // make box its 'left' d-times bisected half
                getDepth<N>(box) + depth, // increment Depth
                getFlags<N>(box));
}

template<Depth N>
__host__ __device__
inline Box<N> decrementDepthBy(const Box<N>& box, Depth depth)
{
  depth = thrust::minimum<Depth>()(depth, getDepth<N>(box));
  
  return Box<N>(safeRightShift<N>(getBitPattern<N>(box), depth), // make box its depth-th level parent
                getDepth<N>(box) - depth, // decrement Depth
                getFlags<N>(box));
}

template<Depth N>
__host__ __device__
inline Box<N> decrementDepthTo(const Box<N>& box, Depth depth)
{
  Box<N> res(box);
  
  // if depth == getLeafDepth() or >= getDepth<N>(box), nothing to do
  if (depth < getDepth<N>(box)) {
    setBitPattern<N>(res) = safeRightShift<N>(getBitPattern<N>(box), getDepth<N>(box) - depth);
    setDepth<N>(res) = depth;
  } else if (depth == getUnsubdividableDepth()) {
    setBitPattern<N>(res) = safeRightShift<N>(getBitPattern<N>(box), 1);
    setDepth<N>(res) -= getDepth<N>(box) > Depth(0) ? Depth(1) : Depth(0);
  }
  
  return res;
}

template<Depth N>
__host__ __device__
inline Box<N> changeDepthTo(const Box<N>& box, Depth depth)
{
  Box<N> res(box);
  
  // for case depth == getDepth(box) or depth == getLeafDepth() nothing to do
  if (depth == getUnsubdividableDepth()) {
    setBitPattern<N>(res) = safeRightShift<N>(getBitPattern<N>(box), 1);
    setDepth<N>(res) -= getDepth<N>(box) > Depth(0) ? Depth(1) : Depth(0);
  } else if (depth < getDepth<N>(box)) {
    setBitPattern<N>(res) = safeRightShift<N>(getBitPattern<N>(box), getDepth<N>(box) - depth);
    setDepth<N>(res) = depth;
  } else if (depth > getDepth<N>(box) && depth != getLeafDepth()) {
    depth = thrust::minimum<Depth>()(depth, getMaxDepth<N>());
    setBitPattern<N>(res) = safeLeftShift<N>(getBitPattern<N>(box), depth - getDepth<N>(box));
    setDepth<N>(res) = depth;
  }
  
  return res;
}

template<Depth N>
__host__ __device__
inline bool hasAtLeastDepth(const Box<N>& box, Depth depth)
{
  return getDepth<N>(box) >= depth || depth == getUnsubdividableDepth() || depth == getLeafDepth();
}

template<Depth N>
__host__ __device__
inline Depth computeDepthOfMutualAncestor(const Box<N>& box1, const Box<N>& box2)
{
  // "map" boxes to getMaxDepth<N>() and determine the number of equal leftmost bits,
  // i.e. the number of leading zeros after XOR
  
  // but the result may not be greater as the depths of both boxes,
  // otherwise, e.g., computeDepthOfMutualAncestor(box,box) == getMaxDepth<N>() != getDepth(box)
  
  thrust::minimum<Depth> min;
  
  return min(countLeadingZeros(safeLeftShift<N>(getBitPattern<N>(box1), N - getDepth<N>(box1))
                               ^ 
                               safeLeftShift<N>(getBitPattern<N>(box2), N - getDepth<N>(box2))),
             min(getDepth<N>(box1), getDepth<N>(box2)));
}

template<Depth N>
__host__ __device__
inline bool isRightHalf(const Box<N>& box)
{
  return getBitPattern<N>(box) & BitPattern<N>(1);
}

template<Depth N>
__host__ __device__
inline Box<N> computeOtherHalf(const Box<N>& box)
{
  return Box<N>(getBitPattern<N>(box) ^ BitPattern<N>(!isSurroundingBox<N>(box) && !isInvalidBox<N>(box)),
                getDepth<N>(box),
                getFlags<N>(box));
}


template<Depth N>
inline std::ostream& operator<<(std::ostream& os, const Box<N>& box)
{
  return os << getBitPattern<N>(box) << ",\t" << getDepth<N>(box) << ",\t" << getFlags<N>(box);
}

} // namespace b12
