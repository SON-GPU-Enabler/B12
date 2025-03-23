
#include "mathFunctions.h"


namespace b12 {

template<Dimension DIM, typename REAL>
__host__ __device__
inline bool isPointInBox(const REAL * point, const REAL * center, const REAL * radius)
{
  bool res = true;
  
#pragma unroll
  for (Dimension j = 0; j < DIM; ++j) {
    res = res && center[j] - radius[j] <= point[j] && point[j] < center[j] + radius[j];
  }
  
  return res;
}

template<Dimension DIM, typename REAL, Depth N>
__host__ __device__
inline Box<N> computeBoxFromPoint(const REAL * point,
                                  const REAL * center,
                                  const REAL * radius,
                                  const Dimension * sdScheme,
                                  const Depth * sdCount,
                                  Flags flags)
{
  Box<N> res(BitPattern<N>(0), getMaxDepth<N>(), flags);
  
  const Depth * _sdCount = sdCount + int(getMaxDepth<N>()) * DIM;
  
  if (isPointInBox<DIM, REAL>(point, center, radius)) {
    // bits[i] = bit pattern in coordinate i
    BitPattern<N> bits[DIM];
    
#pragma unroll
    for (Dimension j = 0; j < DIM; ++j) {
      // bits = (int) (p-c+r)/(2*r) * 2^bisections = (int) ((p-c)/r+1)/2 * 2^bisections
      bits[j] = floorreal(ldexpreal(((point[j] - center[j]) / radius[j] + REAL(1.0)) / REAL(2.0), _sdCount[j]));
    }
    
    // res = combined bits[i] w.r.t. sdScheme
    // unrolling this loop is significantly slower
    for (int i = getMaxDepth<N>() - 1; i >= 0; --i) {
#pragma unroll
      for (Dimension j = 0; j < DIM; ++j) {
        bool temp = sdScheme[i] == j;
        setBitPattern<N>(res) |= safeLeftShift<N>(BitPattern<N>(temp) & bits[j], int(getMaxDepth<N>()) - 1 - i);
        bits[j] = safeRightShift<N>(bits[j], temp);
      }
    }
  } else {
    res = getInvalidBox<N>();
  }
  
  return res;
}

template<Dimension DIM, typename REAL, Depth N>
__host__ __device__
inline Box<N> computeBoxFromPointFixedSdScheme(const REAL * point,
                                               const REAL * center,
                                               const REAL * radius,
                                               Flags flags)
{
  Box<N> res(BitPattern<N>(0), getMaxDepth<N>(), flags);
  
  if (isPointInBox<DIM, REAL>(point, center, radius)) {
#pragma unroll
    for (Dimension j = 0; j < DIM; ++j) {
      
      // bits = (int) (p-c+r)/(2*r) * 2^bisections = (int) ((p-c)/r+1)/2 * 2^bisections
      BitPattern<N> bits = floorreal(ldexpreal(((point[j] - center[j]) / radius[j] + REAL(1.0)) / REAL(2.0),
                                               getMaxDepth<N>() / DIM + Depth(j < getMaxDepth<N>() % DIM)));
      
      int initialShift = (int(getMaxDepth<N>()) - 1 - int(j)) % int(DIM);
      
      // unrolling this loop is significantly slower
      for (int i = 0; i * int(DIM) + initialShift < getMaxDepth<N>(); ++i) {
        // the i-th bit of bits gets stored in the (i * DIM + initialShift)-th bit of res' BitPattern
        // or: leftshift bits to correct position by (i * DIM + initialShift) - i and then get the corresponding bit
        setBitPattern<N>(res) |= safeLeftShift<N>(bits, i * (int(DIM) - 1) + initialShift) & 
                                 safeLeftShift<N>(BitPattern<N>(1), i * int(DIM) + initialShift);
      }
    }
  } else {
    res = getInvalidBox<N>();
  }
  
  return res;
}

template<Dimension DIM, typename REAL, Depth N>
__host__ __device__
inline void computeCoordinatesOfBox(const Box<N>& box,
                                    const REAL * center, 
                                    const REAL * radius,
                                    const Dimension * sdScheme,
                                    const Depth * sdCount,
                                    REAL * res)
{
  BitPattern<N> bits[DIM] = {};
  
  const Depth * _sdCount = sdCount + int(getDepth<N>(box)) * DIM;
  
  for (int i = 0; i < getDepth<N>(box); ++i) {
#pragma unroll
    for (Dimension j = 0; j < DIM; ++j) {
      bool temp = sdScheme[i] == j;
      bits[j] = safeLeftShift<N>(bits[j], temp) |
                BitPattern<N>(temp) & safeRightShift<N>(getBitPattern<N>(box), int(getDepth<N>(box)) - 1 - i);
    }
  }
  
  // p - r_box = REAL(bits) / 2^bisections * 2*r - r + c = (REAL(bits) / 2^bisections * 2 - 1)*r + c
#pragma unroll
  for (Dimension j = 0; j < DIM; ++j) {
    res[j + DIM] = ldexpreal(radius[j], -int(_sdCount[j]));
    res[j] = (ldexpreal(REAL(bits[j]), -int(_sdCount[j])) * REAL(2.0) - REAL(1.0)) * radius[j] + 
             center[j] + res[j + DIM];
  }
  
  res[2 * DIM] = getFlags<N>(box);
}

template<Dimension DIM, typename REAL, Depth N>
__host__ __device__
inline void computeCoordinatesOfBoxFixedSdScheme(const Box<N>& box,
                                                 const REAL * center, 
                                                 const REAL * radius,
                                                 REAL * res)
{
  // p - r_box = REAL(bits) / 2^bisections * 2*r - r + c = (REAL(bits) / 2^bisections * 2 - 1)*r + c
#pragma unroll
  for (Dimension j = 0; j < DIM; ++j) {
    
    BitPattern<N> bits(0);
    int initialShift = (int(DIM) + int(getDepth<N>(box)) - 1 - int(j)) % int(DIM);
    
    // Note: for i*DIM + initialShift >= getDepth(box), the RHS in the loop is zero:
#pragma unroll
    for (int i = 0; i * DIM < getMaxDepth<N>(); ++i) {
      // the (i*DIM+initialShift)-th bit of the box' BitPattern gets stored in the i-th bit
      // of the BitPattern of the resulting point in dimension j,
      // or: shift the BitPattern right to the correct position by (i*DIM+initialShift)-i and
      // then get the corresponding bit
      bits |= safeRightShift<N>(getBitPattern<N>(box), i * (int(DIM) - 1) + initialShift) &
              safeLeftShift<N>(BitPattern<N>(1), i);
    }
    res[j + DIM] = ldexpreal(radius[j], -int(getDepth<N>(box) / DIM + Depth(j < getDepth<N>(box) % DIM)));
    
    res[j] = (ldexpreal(REAL(bits), -int(getDepth<N>(box) / DIM + Depth(j < getDepth<N>(box) % DIM))) * REAL(2.0) - 
              REAL(1.0)) * radius[j] + center[j] + res[j + DIM];
  }
  
  res[2 * DIM] = getFlags<N>(box);
}

} // namespace b12
