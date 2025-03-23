
namespace b12 {

// important condition: getMaxDepth<N>() < getUnsubdividableDepth() < getLeafDepth() !!!
__host__ __device__
inline Depth getLeafDepth()
{
  return Depth(-1);
}

__host__ __device__
inline Depth getUnsubdividableDepth()
{
  return Depth(-2);
}

template<Depth N>
__host__ __device__
inline Depth getMaxDepth()
{
  return N >= Depth(-2) ? Depth(-3) : N; // == min{N, Depth_MAX - 2}
}

// important condition: getMaxNrBoxes() < getInvalidNrBoxes() !!!
__host__ __device__
inline NrBoxes getInvalidNrBoxes()
{
  return NrBoxes(-1); // == min{BitPattern_MAX, NrBoxes_MAX}
}

__host__ __device__
inline NrBoxes getMaxNrBoxes()
{
  return NrBoxes(-2); // == min{BitPattern_MAX, NrBoxes_MAX - 1}
}

} // namespace b12
