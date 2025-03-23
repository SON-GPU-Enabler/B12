
#pragma once

#include <iostream>

#include <thrust/tuple.h>

#include "BitPattern.h"
#include "Flags.h"
#include "TypeDefinitions.h"


namespace b12 {

// definition of a Box
template<Depth N>
using Box = thrust::tuple<BitPattern<N>, Depth, Flags>;


// the box representing the surroundig/outer box of a BoxCollection
template<Depth N>
__host__ __device__
Box<N> getSurroundingBox();

// the box that would be always past-the-end in ordering, see also isStrictlyPreceding
template<Depth N>
__host__ __device__
Box<N> getInvalidBox();


/// Reference functions -----------------------------------------------------------------------------------------------

template<Depth N>
__host__ __device__
const BitPattern<N>& getBitPattern(const Box<N>& box);

template<Depth N>
__host__ __device__
BitPattern<N>& setBitPattern(Box<N>& box);

template<Depth N>
__host__ __device__
const Depth& getDepth(const Box<N>& box);

template<Depth N>
__host__ __device__
Depth& setDepth(Box<N>& box);

template<Depth N>
__host__ __device__
const Flags& getFlags(const Box<N>& box);

template<Depth N>
__host__ __device__
Flags& setFlags(Box<N>& box);


/// comparison operations ---------------------------------------------------------------------------------------------

// Idea: get Box's BitPattern in Depth getMaxDepth<N>(), i.e.
// (BitPattern*2+1)*2^(getMaxDepth<N>()-Depth-1)-1
//   == BitPattern*2^(getMaxDepth<N>()-Depth) + 2^(getMaxDepth<N>()-Depth-1) - 1
//   == [BitPattern]01111...1
//   == Box with largest BitPattern in Depth getMaxDepth<N>() whose BitPattern <= the Box' BitPattern
// in the special case that a box is mapped onto the other one, the Depths are compared
template<Depth N>
__host__ __device__
BitPattern<N> getComparableBitPattern(const Box<N>& box);

template<Depth N>
__host__ __device__
bool hasSamePath(const Box<N>& box1, const Box<N>& box2); // is true if one is a subset of the other one

template<Depth N>
__host__ __device__
bool isInvalidBox(const Box<N>& box);

template<Depth N>
__host__ __device__
bool isSurroundingBox(const Box<N>& box);

template<Depth N>
__host__ __device__
bool isPrecedingInMinorDepth(const Box<N>& box1, const Box<N>& box2);
// no Strict Weak Ordering: if y is the parent of x and z, then:
// x !< y && y !< z should imply x !< z but x < z holds (in other words: x == y && y == z but x < z)

template<Depth N>
__host__ __device__
bool isSameBox(const Box<N>& box1, const Box<N>& box2); // independent of Flags

// Invalid Box always succeeds valid boxes
template<Depth N>
__host__ __device__
bool isStrictlyPreceding(const Box<N>& box1, const Box<N>& box2);


/// depth operations --------------------------------------------------------------------------------------------------

// Resulting box is the 'left' half of the depth-times bisected input box
// undefined behaviour for negative depth
template<Depth N>
__host__ __device__
Box<N> incrementDepthBy(const Box<N>& box, Depth depth);

// Resulting box is the hierachically depth-th parent of the input box
// undefined behaviour for negative depth
template<Depth N>
__host__ __device__
Box<N> decrementDepthBy(const Box<N>& box, Depth depth);

// Resulting box is the corresponding box on Depth depth in the same path in the Tree
// if depth > depth_of_box, no changes happen
template<Depth N>
__host__ __device__
Box<N> decrementDepthTo(const Box<N>& box, Depth depth);

// Resulting box is the corresponding box on Depth depth in the same path in the Tree
template<Depth N>
__host__ __device__
Box<N> changeDepthTo(const Box<N>& box, Depth depth);

// checks whether a box has at least Depth depth
template<Depth N>
__host__ __device__
bool hasAtLeastDepth(const Box<N>& box, Depth depth);

// computes the maximal depth in which both boxes are equal; undefined behaviour for invalid boxes
template<Depth N>
__host__ __device__
Depth computeDepthOfMutualAncestor(const Box<N>& box1, const Box<N>& box2);


/// "tree" operations -------------------------------------------------------------------------------------------------

template<Depth N>
__host__ __device__
bool isRightHalf(const Box<N>& box);

// Resulting box is the other half of the box containing the input box as one half
// Surrounding Box and Invalid Box have no other half
template<Depth N>
__host__ __device__
Box<N> computeOtherHalf(const Box<N>& box);

template<Depth N>
std::ostream& operator<<(std::ostream& os, const Box<N>& box);

} // namespace b12


#include "Box.hpp"
