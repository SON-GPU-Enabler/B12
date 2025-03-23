
namespace b12 {

template<Dimension DIM, typename REAL, typename MAP>
__host__ __device__
inline IteratingMap<DIM, REAL, MAP>::IteratingMap(MAP map, uint32_t steps) : map_(map), steps_(steps) {}

template<Dimension DIM, typename REAL, typename MAP>
__host__ __device__
inline bool IteratingMap<DIM, REAL, MAP>::operator()(const REAL * point, REAL * res) const
{
  uint32_t i = 0;
  
  // if steps_ is odd, apply the map once, otherwise copy point to res
  if (steps_ & uint32_t(1)) {
    map_(point, res);
    ++i;
  } else {
#pragma unroll
    for (Dimension j = 0; j < DIM; ++j) {
      res[j] = point[j]; // indices are constant/known at compile time
    }
  }
  
  for (; i < steps_; i += uint32_t(2)) {
    REAL temp[DIM];
    map_(res, temp);
    map_(temp, res);
  }
  
  return false;
}


template<Dimension DIM, typename REAL, typename RHS>
__host__ __device__
inline RK4<DIM, REAL, RHS>::RK4(RHS rightHandSide, REAL t0, REAL tEnd, REAL h)
    : rightHandSide_(rightHandSide), t0_(t0), tEnd_(tEnd), h_(h) {}

template<Dimension DIM, typename REAL, typename RHS>
__host__ __device__
inline bool RK4<DIM, REAL, RHS>::operator()(const REAL * point, REAL * res) const
{
  REAL coords[3 * DIM];
  // coords[0       ...     DIM - 1] == point
  // coords[DIM     ... 2 * DIM - 1] == k_out
  // coords[2 * DIM ... 3 * DIM - 1] == k_arg
  
#pragma unroll
  for (int j = 0; j < DIM; ++j) {
    coords[j] = point[j];
  }
  
  for (REAL t = t0_; t < tEnd_; t += h_) {
    // perhaps adapt h_
    REAL h = thrust::minimum<REAL>()(h_, tEnd_ - t);
    // k_out == k_1
    rightHandSide_(t, coords, coords + DIM);
    // res = point + h/6 * k_1;
#pragma unroll
    for (int j = 0; j < DIM; ++j) {
      res[j] = coords[j] + h / REAL(6.0) * coords[DIM + j];
    }
    // k_arg = point + h/2 * k_1;
#pragma unroll
    for (int j = 0; j < DIM; ++j) {
      coords[2 * DIM + j] = coords[j] + h / REAL(2.0) * coords[DIM + j];
    }
    // k_out == k_2
    rightHandSide_(t + h / REAL(2.0), coords + 2 * DIM, coords + DIM);
    // res += h/3 * k_2;
#pragma unroll
    for (int j = 0; j < DIM; ++j) {
      res[j] += h / REAL(3.0) * coords[DIM + j];
    }
    // k_arg = point + h/2 * k_2;
#pragma unroll
    for (int j = 0; j < DIM; ++j) {
      coords[2 * DIM + j] = coords[j] + h / REAL(2.0) * coords[DIM + j];
    }
    // k_out == k_3
    rightHandSide_(t + h / REAL(2.0), coords + 2 * DIM, coords + DIM);
    // res += h/3 * k_3;
#pragma unroll
    for (int j = 0; j < DIM; ++j) {
      res[j] += h / REAL(3.0) * coords[DIM + j];
    }
    // k_arg = point + h * k_3;
#pragma unroll
    for (int j = 0; j < DIM; ++j) {
      coords[2 * DIM + j] = coords[j] + h * coords[DIM + j];
    }
    // k_out == k_4
    rightHandSide_(t + h, coords + 2 * DIM, coords + DIM);
    // res += h/6 * k_4;
#pragma unroll
    for (int j = 0; j < DIM; ++j) {
      res[j] += h / REAL(6.0) * coords[DIM + j];
    }
    // point = res; (for further iterations)
#pragma unroll
    for (int j = 0; j < DIM; ++j) {
      coords[j] = res[j];
    }
  }
  
  return false;
}


template<Dimension DIM, typename REAL, typename RHS>
__host__ __device__
inline RK4Auto<DIM, REAL, RHS>::RK4Auto(RHS rightHandSide, REAL h, uint32_t steps)
    : rightHandSide_(rightHandSide), h_(h), steps_(steps) {}

template<Dimension DIM, typename REAL, typename RHS>
__host__ __device__
inline bool RK4Auto<DIM, REAL, RHS>::operator()(const REAL * point, REAL * res) const
{
  REAL coords[3 * DIM];
  // coords[0       ...     DIM - 1] == point
  // coords[DIM     ... 2 * DIM - 1] == k_out
  // coords[2 * DIM ... 3 * DIM - 1] == k_arg
  
#pragma unroll
  for (int j = 0; j < DIM; ++j) {
    coords[j] = point[j];
  }
  
  for (uint32_t i = 0; i < steps_; ++i) {
    // k_out == k_1
    rightHandSide_(coords, coords + DIM);
    // res = point + h/6 * k_1;
#pragma unroll
    for (int j = 0; j < DIM; ++j) {
      res[j] = coords[j] + h_ / REAL(6.0) * coords[DIM + j];
    }
    // k_arg = point + h/2*k_1;
#pragma unroll
    for (int j = 0; j < DIM; ++j) {
      coords[2 * DIM + j] = coords[j] + h_ / REAL(2.0) * coords[DIM + j];
    }
    // k_out == k_2
    rightHandSide_(coords + 2 * DIM, coords + DIM);
    // res += h/3 * k_2;
#pragma unroll
    for (int j = 0; j < DIM; ++j) {
      res[j] += h_ / REAL(3.0) * coords[DIM + j];
    }
    // k_arg = point + h/2*k_2;
#pragma unroll
    for (int j = 0; j < DIM; ++j) {
      coords[2 * DIM + j] = coords[j] + h_ / REAL(2.0) * coords[DIM + j];
    }
    // k_out == k_3
    rightHandSide_(coords + 2 * DIM, coords + DIM);
    // res += h/3 * k_3;
#pragma unroll
    for (int j = 0; j < DIM; ++j) {
      res[j] += h_ / REAL(3.0) * coords[DIM + j];
    }
    // k_arg = point + h * k_3;
#pragma unroll
    for (int j = 0; j < DIM; ++j) {
      coords[2 * DIM + j] = coords[j] + h_ * coords[DIM + j];
    }
    // k_out == k_4
    rightHandSide_(coords + 2 * DIM, coords + DIM);
    // res += h/6 * k_4;
#pragma unroll
    for (int j = 0; j < DIM; ++j) {
      res[j] += h_ / REAL(6.0) * coords[DIM + j];
    }
    // point = res; (for further iterations)
#pragma unroll
    for (int j = 0; j < DIM; ++j) {
      coords[j] = res[j];
    }
  }
  
  return false;
}


template<Dimension DIM, typename REAL, typename RHS1, typename RHS2>
__host__ __device__
inline SymplecticIntegrator<DIM, REAL, RHS1, RHS2>::SymplecticIntegrator(uint8_t order, 
                                                                         RHS1 rightHandSide1, RHS2 rightHandSide2,
                                                                         REAL t0, REAL tEnd, REAL h)
    : order_(order), rightHandSide1_(rightHandSide1), rightHandSide2_(rightHandSide2), t0_(t0), tEnd_(tEnd), h_(h) {}

template<Dimension DIM, typename REAL, typename RHS1, typename RHS2>
__host__ __device__
inline bool SymplecticIntegrator<DIM, REAL, RHS1, RHS2>::operator()(const REAL * point, REAL * res) const
{
  REAL temp[DIM / 2];
  
#pragma unroll
  for (int j = 0; j < DIM; ++j) {
    res[j] = point[j];
  }
  
  if (order_ == 1) {
    
    for (REAL t = t0_; t < tEnd_; t += h_) {
      // perhaps adapt h_
      REAL h = thrust::minimum<REAL>()(h_, tEnd_ - t);
      
      // p = p + RHS2(t, q) * h;
      rightHandSide2_(t, res, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[DIM / 2 + j] += temp[j] * h;
      }
      // q = q + RHS1(t, p) * h;
      rightHandSide1_(t, res + DIM / 2, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j] += temp[j] * h;
      }
    }
    
  } else if (order_ == 2) {
    
    for (REAL t = t0_; t < tEnd_; t += h_) {
      // perhaps adapt h_
      REAL h = thrust::minimum<REAL>()(h_, tEnd_ - t);
      
      // p = p + 0 * RHS2(t, q) * h;
      // --> first step vanishes
      // q = q + 0.5 * RHS1(t, p) * h;
      rightHandSide1_(t, res + DIM / 2, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j] += REAL(0.5) * temp[j] * h;
      }
      
      // p = p + 1 * RHS2(t, q) * h;
      rightHandSide2_(t, res, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j + DIM / 2] += temp[j] * h;
      }
      // q = q + 0.5 * RHS1(t, p) * h;
      rightHandSide1_(t, res + DIM / 2, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j] += REAL(0.5) * temp[j] * h;
      }
    }
    
  } else if (order_ == 3) {
    
    for (REAL t = t0_; t < tEnd_; t += h_) {
      // perhaps adapt h_
      REAL h = thrust::minimum<REAL>()(h_, tEnd_ - t);
      
      // p = p + 1 * RHS2(t, q) * h;
      rightHandSide2_(t, res, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[DIM / 2 + j] += temp[j] * h;
      }
      // q = q - 1/24 * RHS1(t, p) * h;
      rightHandSide1_(t, res + DIM / 2, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j] -= REAL(1.0) / REAL(24.0) * temp[j] * h;
      }
      
      // p = p - 2/3 * RHS2(t, q) * h;
      rightHandSide2_(t, res, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j + DIM / 2] -= REAL(2.0) / REAL(3.0) * temp[j] * h;
      }
      // q = q + 3/4 * RHS1(t, p) * h;
      rightHandSide1_(t, res + DIM / 2, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j] += REAL(3.0) / REAL(4.0) * temp[j] * h;
      }
      
      // p = p + 2/3 * RHS2(t, q) * h;
      rightHandSide2_(t, res, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j + DIM / 2] += REAL(2.0) / REAL(3.0) * temp[j] * h;
      }
      // q = q + 7/24 * RHS1(t, p) * h;
      rightHandSide1_(t, res + DIM / 2, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j] += REAL(7.0) / REAL(24.0) * temp[j] * h;
      }
    }
    
  } else if (order_ == 4) {
    
    REAL cbrt2 = cbrtreal(REAL(2.0));
    REAL factor = REAL(1.0) / (REAL(2.0) - cbrt2);
    
    for (REAL t = t0_; t < tEnd_; t += h_) {
      // perhaps adapt h_
      REAL h = thrust::minimum<REAL>()(h_, tEnd_ - t);
      
      // p = p + factor/2 * RHS2(t, q) * h;
      rightHandSide2_(t, res, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[DIM / 2 + j] += REAL(0.5) * factor * temp[j] * h;
      }
      // q = q + factor * RHS1(t, p) * h;
      rightHandSide1_(t, res + DIM / 2, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j] += factor * temp[j] * h;
      }
      
      // p = p + (1 - cbrt2)/2 * factor * RHS2(t, q) * h;
      rightHandSide2_(t, res, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j + DIM / 2] += REAL(0.5) * (REAL(1.0) - cbrt2) * factor * temp[j] * h;
      }
      // q = q - cbrt2 * factor * RHS1(t, p) * h;
      rightHandSide1_(t, res + DIM / 2, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j] -= cbrt2 * factor * temp[j] * h;
      }
      
      // p = p + (1 - cbrt2)/2 * factor * RHS2(t, q) * h;
      rightHandSide2_(t, res, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j + DIM / 2] += REAL(0.5) * (REAL(1.0) - cbrt2) * factor * temp[j] * h;
      }
      // q = q + factor * RHS1(t, p) * h;
      rightHandSide1_(t, res + DIM / 2, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j] += factor * temp[j] * h;
      }
      
      // p = p + factor/2 * RHS2(t, q) * h;
      rightHandSide2_(t, res, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j + DIM / 2] += REAL(0.5) * factor * temp[j] * h;
      }
      // q = q + 0 * RHS1(t, p) * h;
      // --> last step vanishes
    }
  }
    
  return false;
}


template<Dimension DIM, typename REAL, typename RHS1, typename RHS2>
__host__ __device__
inline SymplecticIntegratorAuto<DIM, REAL, RHS1, RHS2>::SymplecticIntegratorAuto(uint8_t order, 
                                                                                 RHS1 rightHandSide1,
                                                                                 RHS2 rightHandSide2,
                                                                                 REAL h, uint32_t steps)
    : order_(order), rightHandSide1_(rightHandSide1), rightHandSide2_(rightHandSide2), h_(h), steps_(steps) {}

template<Dimension DIM, typename REAL, typename RHS1, typename RHS2>
__host__ __device__
inline bool SymplecticIntegratorAuto<DIM, REAL, RHS1, RHS2>::operator()(const REAL * point, REAL * res) const
{
  REAL temp[DIM / 2];
  
#pragma unroll
  for (int j = 0; j < DIM; ++j) {
    res[j] = point[j];
  }
  
  if (order_ == 1) {
    
    for (uint32_t i = 0; i < steps_; ++i) {
      // p = p + RHS2(q) * h_;
      rightHandSide2_(res, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[DIM / 2 + j] += temp[j] * h_;
      }
      // q = q + RHS1(p) * h_;
      rightHandSide1_(res + DIM / 2, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j] += temp[j] * h_;
      }
    }
    
  } else if (order_ == 2) {
    
    for (uint32_t i = 0; i < steps_; ++i) {
      // p = p + 0 * RHS2(q) * h_;
      // --> first step vanishes
      // q = q + 0.5 * RHS1(p) * h_;
      rightHandSide1_(res + DIM / 2, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j] += REAL(0.5) * temp[j] * h_;
      }
      
      // p = p + 1 * RHS2(q) * h_;
      rightHandSide2_(res, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j + DIM / 2] += temp[j] * h_;
      }
      // q = q + 0.5 * RHS1(p) * h_;
      rightHandSide1_(res + DIM / 2, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j] += REAL(0.5) * temp[j] * h_;
      }
    }
    
  } else if (order_ == 3) {
    
    for (uint32_t i = 0; i < steps_; ++i) {
      // p = p + 1 * RHS2(q) * h_;
      rightHandSide2_(res, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[DIM / 2 + j] += temp[j] * h_;
      }
      // q = q - 1/24 * RHS1(p) * h_;
      rightHandSide1_(res + DIM / 2, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j] -= REAL(1.0) / REAL(24.0) * temp[j] * h_;
      }
      
      // p = p - 2/3 * RHS2(q) * h_;
      rightHandSide2_(res, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j + DIM / 2] -= REAL(2.0) / REAL(3.0) * temp[j] * h_;
      }
      // q = q + 3/4 * RHS1(p) * h_;
      rightHandSide1_(res + DIM / 2, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j] += REAL(3.0) / REAL(4.0) * temp[j] * h_;
      }
      
      // p = p + 2/3 * RHS2(q) * h_;
      rightHandSide2_(res, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j + DIM / 2] += REAL(2.0) / REAL(3.0) * temp[j] * h_;
      }
      // q = q + 7/24 * RHS1(p) * h_;
      rightHandSide1_(res + DIM / 2, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j] += REAL(7.0) / REAL(24.0) * temp[j] * h_;
      }
    }
    
  } else if (order_ == 4) {
    
    REAL cbrt2 = cbrtreal(REAL(2.0));
    REAL factor = REAL(1.0) / (REAL(2.0) - cbrt2);
    
    for (uint32_t i = 0; i < steps_; ++i) {
      // p = p + factor/2 * RHS2(q) * h_;
      rightHandSide2_(res, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[DIM / 2 + j] += REAL(0.5) * factor * temp[j] * h_;
      }
      // q = q + factor * RHS1(p) * h_;
      rightHandSide1_(res + DIM / 2, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j] += factor * temp[j] * h_;
      }
      
      // p = p + (1 - cbrt2)/2 * factor * RHS2(q) * h_;
      rightHandSide2_(res, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j + DIM / 2] += REAL(0.5) * (REAL(1.0) - cbrt2) * factor * temp[j] * h_;
      }
      // q = q - cbrt2 * factor * RHS1(p) * h_;
      rightHandSide1_(res + DIM / 2, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j] -= cbrt2 * factor * temp[j] * h_;
      }
      
      // p = p + (1 - cbrt2)/2 * factor * RHS2(q) * h_;
      rightHandSide2_(res, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j + DIM / 2] += REAL(0.5) * (REAL(1.0) - cbrt2) * factor * temp[j] * h_;
      }
      // q = q + factor * RHS1(p) * h_;
      rightHandSide1_(res + DIM / 2, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j] += factor * temp[j] * h_;
      }
      
      // p = p + factor/2 * RHS2(q) * h_;
      rightHandSide2_(res, temp);
#pragma unroll
      for (int j = 0; j < DIM / 2; ++j) {
        res[j + DIM / 2] += REAL(0.5) * factor * temp[j] * h_;
      }
      // q = q + 0 * RHS1(p) * h_;
      // --> last step vanishes
    }
  }
    
  return false;
}

} // namespace b12
