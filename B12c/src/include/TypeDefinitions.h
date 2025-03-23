
#pragma once

#include "math.h"
#include <cstdint>


namespace b12 {

// actually only important for saving/reading files
const uint32_t B12_VERSION_MAJOR = 'c' - 'a' + 1;
const uint32_t B12_VERSION_MINOR = 1;


// Depth for "tree";
// must be an unsigned integer whose arithmetic results fit into int;
// largest value "-1" is reserved for depth of leaves;
// second largest value "-2" is reserved for depth of unsubdividable boxes
using Depth = uint8_t; // 0 ... 255

// depth of leaves
__host__ __device__
Depth getLeafDepth();

// depth of unsubdividable boxes
__host__ __device__
Depth getUnsubdividableDepth();

// maximal possible Depth <= N
template<Depth N>
__host__ __device__
Depth getMaxDepth();


// Phase space dimension;
// must be an unsigned integer whose arithmetic results fit into int
using Dimension = uint8_t; // 0 ... 255


// possible number of boxes;
// must be an unsigned integer;
// largest value "-1" is reserved for non-valid (search) index;
// recommendation: std::numeric_limits<NrBoxes>::digits <= 1/2 * std::numeric_limits<NrPoints>::digits
// so that radix sort can be used in transitionMatrix
using NrBoxes = uint32_t; // 0 ... 4,294,967,295 - 1

// important condition: getMaxNrBoxes() < getInvalidNrBoxes() !!!
__host__ __device__
NrBoxes getInvalidNrBoxes();

// maximal number of boxes that can be stored,
__host__ __device__
NrBoxes getMaxNrBoxes();


// possible number of points (preferably largest unsigned integer type)
using NrPoints = uint64_t; // 0 ... 18,446,744,073,709,551,615

} // namespace b12


#include "TypeDefinitions.hpp"
