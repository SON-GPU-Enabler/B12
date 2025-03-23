
namespace b12 {

template<Depth N>
__host__ __device__
inline BitPattern<N> safeLeftShift(const BitPattern<N>& b, int i)
{
  if (i <= 0) {
    return b;
  } else if (i >= N) {
    return BitPattern<N>(0);
  } else {
    return b << i;
  }
}

template<Depth N>
__host__ __device__
inline BitPattern<N> safeRightShift(const BitPattern<N>& b, int i)
{
  if (i <= 0) {
    return b;
  } else if (i >= N) {
    return BitPattern<N>(0);
  } else {
    return b >> i;
  }
}

} // namespace b12
