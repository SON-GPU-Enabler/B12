
#pragma once

#include "BitPattern.h"
#include "Box.h"
#include "Flags.h"
#include "TypeDefinitions.h"


namespace b12 {

// checks whether the point lies in the box that is determined by center and radius
template<Dimension DIM, typename REAL>
__host__ __device__
bool isPointInBox(const REAL * point,
                  const REAL * center,
                  const REAL * radius);

template<Dimension DIM, typename REAL, Depth N>
__host__ __device__
Box<N> computeBoxFromPoint(const REAL * point,
                           const REAL * center,
                           const REAL * radius,
                           const Dimension * sdScheme,
                           const Depth * sdCount,
                           Flags flags);

template<Dimension DIM, typename REAL, Depth N>
__host__ __device__
Box<N> computeBoxFromPointFixedSdScheme(const REAL * point,
                                        const REAL * center,
                                        const REAL * radius,
                                        Flags flags);

// computes the coordinates (center, radius, flags) of a Box and stores them in res
// i.e. res should point to a REAL range of at least 2*DIM+1 elements
template<Dimension DIM, typename REAL, Depth N>
__host__ __device__
void computeCoordinatesOfBox(const Box<N>& box,
                             const REAL * center, 
                             const REAL * radius,
                             const Dimension * sdScheme,
                             const Depth * sdCount,
                             REAL * res);

template<Dimension DIM, typename REAL, Depth N>
__host__ __device__
void computeCoordinatesOfBoxFixedSdScheme(const Box<N>& box,
                                          const REAL * center, 
                                          const REAL * radius,
                                          REAL * res);

} // namespace b12


#include "coordinateFunctions.hpp"
