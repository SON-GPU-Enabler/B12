
#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>

#include "BitPattern.h"
#include "Box.h"
#include "CooMatrix.h"
#include "Flags.h"
#include "Functors.h"
#include "helpFunctions.h"
#include "ThrustSystem.h"
#include "TypeDefinitions.h"


namespace b12 {

// A determines storage location and parallelization scheme, DIM the dimension of the boxes,
// REAL the floating-point precision, N the maximal possible depth
template<Architecture A, Dimension DIM, typename REAL = double, Depth N = 64>
class ImplicitBoxTree
{
 public:
  
  template<Architecture A2, Dimension DIM2, typename REAL2, Depth N2>
  friend class ImplicitBoxTree;
  
  
  /// basic class functions: constructors, destructors, copy assignment, and hard-disk store functions ----------------
  
  // default constructor defines the unit hypercube
  ImplicitBoxTree();
  
  template<Architecture A2, Dimension DIM2, typename REAL2, Depth N2>
  ImplicitBoxTree(const ImplicitBoxTree<A2, DIM2, REAL2, N2>& implicitBoxTree);
  
  template<typename REAL2>
  ImplicitBoxTree(const typename ThrustSystem<A>::Vector<REAL2>& center,
                  const typename ThrustSystem<A>::Vector<REAL2>& radius);
  
  template<typename REAL2, typename UInt>
  ImplicitBoxTree(const typename ThrustSystem<A>::Vector<REAL2>& center,
                  const typename ThrustSystem<A>::Vector<REAL2>& radius,
                  const typename ThrustSystem<A>::Vector<UInt>& sdScheme);
  
  /// loads implicit box tree that was saved into file 'fileName'
  ImplicitBoxTree(const std::string& fileName);
  
  
  /// automatically compiled destructor:
  // ~ImplicitBoxTree();
  
  
  template<Architecture A2, Dimension DIM2, typename REAL2, Depth N2>
  ImplicitBoxTree<A, DIM, REAL, N>& operator=(const ImplicitBoxTree<A2, DIM2, REAL2, N2>& implicitBoxTree);
  
  
  /// stores the object on hard-disk in file 'fileName' overwriting previously existing data in that file;
  /// if fileName does not have the suffix ".json", that will be appended
  void save(const std::string& fileName) const;
  
  /// stores the object on hard-disk in file 'fileName' such that it is usable in readForGaio.m
  /// overwriting previously existing data in that file
  void saveForGaio(const std::string& fileName) const;
  
  
  /// "tree" functions ------------------------------------------------------------------------------------------------
  
  /// computes sdCount_ from sdScheme_
  void adaptSdCount();
  
  /// checks whether all leaves lie a given Depth depth
  bool areAllLeavesOnDepth(Depth depth) const;
  
  /// stores the boxes of Depth depth successively in res, each in the format {center, radius, Flags}
  void boxes(Depth depth, REAL * res_begin) const;
  
  /// returns a vector containing successively the boxes of Depth depth, each in the format {center, radius, Flags}
  typename ThrustSystem<A>::Vector<REAL> boxes(Depth depth) const;
  
  /// see setFlags, but changes the flags from flagsToUnset (if set) to flagsToSet
  NrBoxes changeFlags(Flags flagsToUnset, Flags flagsToSet, Depth depth = getLeafDepth());
  
  template<typename BoxIteratorPair>
  NrBoxes changeFlagsByIterators(BoxIteratorPair points,
                                 Flags flagsToUnset, Flags flagsToSet, Depth depth = getLeafDepth());
  
  NrBoxes changeFlagsByPoints(const typename ThrustSystem<A>::Vector<REAL>& points,
                              Flags flagsToUnset, Flags flagsToSet, Depth depth = getLeafDepth());
  
  NrBoxes changeFlagsBySearch(const typename ThrustSystem<A>::Vector<NrBoxes>& searchVector,
                              Flags flagsToUnset, Flags flagsToSet, Depth depth = getLeafDepth());
  
  template<typename BoolIterator>
  NrBoxes changeFlagsByStencil(BoolIterator stencil_begin,
                               Flags flagsToUnset, Flags flagsToSet, Depth depth = getLeafDepth());
  
  /// returns the number of boxes on Depth depth
  NrBoxes count(Depth depth = getLeafDepth()) const;
  
  /// returns the number of boxes on Depth depth whose Flags fulfill pred;
  /// the Flags of each box are the combination of all its leaf Flags, i.e. Combination can be
  /// FlagsAndFunctor or FlagsOrFunctor
  template<typename Predicate, typename Combination>
  NrBoxes countIfFlags(Predicate pred, Combination comb, Depth depth);
  
  /// returns the number of boxes on Depth depth that have Flags flags set and the corresponding stencil is true,
  /// i.e. whose leaves all have Flags flags set, and the corresponding stencil is true
  template<typename Predicate, typename Combination, typename BoolIterator>
  NrBoxes countIfFlags(Predicate pred, Combination comb, BoolIterator stencil_begin, Depth depth);
  
  /// deletes all boxes on Depth depth so that afterwards the tree has at most Depth max(depth-1, 0);
  /// if depth == 0, all Flags will be removed;
  /// returns the number of deleted boxes on Depth depth
  NrBoxes deleteDepth(Depth depth);
  
  /// returns the current maximal depth of the tree
  Depth depth() const;
  
  /// returns the vector {count(0), …, count(depth())}
  typename ThrustSystem<A>::Vector<NrBoxes> enumerate() const;
  
  /// frees the memory of indicesInDepth_
  void freeIndicesInDepth();
  
  /// frees the memory of leafIndicesFromDepth_
  void freeLeafIndicesFromDepth();
  
  /// checks whether both implicitBoxTree represent the same box collection (inclusing test for Flags)
  template<Architecture A2, Dimension DIM2, typename REAL2, Depth N2>
  bool hasSameBoxes(const ImplicitBoxTree<A2, DIM2, REAL2, N2>& implicitBoxTree) const;
  
  /// initialises indicesInDepth_;
  /// boxes with Depth < depth get the same index as their preceding box with Depth >= depth 
  /// or 0 if no such predecessor exists;
  /// returns a bool-iterator indicating where a "new" box in Depth depth "starts"
  thrust::transform_iterator<
      IsUnequalToPredecessorInDepthFunctor<N>,
      thrust::zip_iterator<
          thrust::tuple<
              thrust::counting_iterator<NrBoxes>,
              thrust::transform_iterator<
                  DecrementDepthToFunctor<N>,
                  typename ThrustSystem<A>::ConstBoxIterator<N>>,
              thrust::transform_iterator<
                  DecrementDepthToFunctor<N>,
                  thrust::permutation_iterator<
                      typename ThrustSystem<A>::ConstBoxIterator<N>,
                      thrust::transform_iterator<
                          DecrementWhenPositiveFunctor,
                          thrust::counting_iterator<NrBoxes>>>>>>>
  initializeIndicesInDepth(Depth depth);
  
  /// initialises leafIndicesFromDepth_ containing all "first" leaf indices 
  /// of boxes in Depth depth that have Flags flagsToCheck set
  void initializeLeafIndicesFromDepth(Depth depth, Flags flagsToCheck = Flag::NONE);
  
  /// inserts the boxes in Depth depth that contain the points given by the range [points.first, points.second);
  /// Depth getLeafDepth() means that for each point a new leaf with minimal Depth is created;
  /// in each newly created box the Flags flagsForInserted are set, in hit boxes the Flags flagsForHit are set;
  /// the resulting range [res_begin, res_begin + (points.second - points.first)) will contain true iff 
  /// for the corresponding point a box was created, otherwise false;
  /// all boxes in [points.first, points.second) are supposed to have the same Flags set and at least Depth getMaxDepth
  template<typename BoxIteratorPair, typename BoolIterator>
  void insertByIterators(BoxIteratorPair points,
                         Depth depth, Flags flagsForInserted, Flags flagsForHit,
                         BoolIterator res_begin);
  
  /// same as void insert(…), but returns a vector containing the resulting range;
  /// default values: flagsForInserted = Flag::INS, flagsForHit = Flag::NONE
  template<typename BoxIteratorPair>
  typename ThrustSystem<A>::Vector<bool> insertByIterators(BoxIteratorPair points,
                                                           Depth depth,
                                                           Flags flagsForInserted = Flag::INS,
                                                           Flags flagsForHit = Flag::NONE);
  
  /// same as void insert(…), but inserts the points given by the
  /// vector {points[0], …, points[DIM-1] ; points[DIM], …, points[2*DIM-1] ; …}
  template<typename BoolIterator>
  void insertByPoints(const typename ThrustSystem<A>::Vector<REAL>& points,
                      Depth depth, Flags flagsForInserted, Flags flagsForHit,
                      BoolIterator res_begin);
  
  /// same as void insert(…), but inserts the points given by the
  /// vector {points[0], …, points[DIM-1] ; points[DIM], …, points[2*DIM-1] ; …};
  /// returns a vector containing the resulting range
  typename ThrustSystem<A>::Vector<bool> insertByPoints(const typename ThrustSystem<A>::Vector<REAL>& points,
                                                        Depth depth,
                                                        Flags flagsForInserted = Flag::INS,
                                                        Flags flagsForHit = Flag::NONE);
  
  /// inserts Surrounding Box into the ImplicitBoxTree if it is empty
  void insertSurroundingBoxIfEmpty();
  
  /// returns a bool-iterator indicating where a "new" box in Depth depth "starts"
  thrust::transform_iterator<
      IsUnequalToPredecessorInDepthFunctor<N>,
      thrust::zip_iterator<
          thrust::tuple<
              thrust::counting_iterator<NrBoxes>,
              thrust::transform_iterator<
                  DecrementDepthToFunctor<N>,
                  typename ThrustSystem<A>::ConstBoxIterator<N>>,
              thrust::transform_iterator<
                  DecrementDepthToFunctor<N>,
                  thrust::permutation_iterator<
                      typename ThrustSystem<A>::ConstBoxIterator<N>,
                      thrust::transform_iterator<
                          DecrementWhenPositiveFunctor,
                          thrust::counting_iterator<NrBoxes>>>>>>>
  isNewBoxInDepth_begin(Depth depth) const;
  
  /// returns a bool-iterator indicating whether a leaf is unsubdividable
  thrust::transform_iterator<
      IsUnsubdividableFunctor<N>,
      thrust::zip_iterator<
          thrust::tuple<
              typename ThrustSystem<A>::ConstBoxIterator<N>,
              thrust::permutation_iterator<
                  typename ThrustSystem<A>::ConstBoxIterator<N>,
                  thrust::transform_iterator<
                      DecrementWhenPositiveFunctor,
                      thrust::counting_iterator<NrBoxes>>>,
              thrust::permutation_iterator<
                  typename ThrustSystem<A>::ConstBoxIterator<N>,
                  thrust::transform_iterator<
                      CapByFunctor,
                      thrust::counting_iterator<NrBoxes>>>>>>
  isUnsubdividable_begin(Flags flags = Flag::NONE) const;
  
  /// checks whether sdScheme_ == {0, 1, …, DIM-1, 0, 1, …, DIM-1, 0, 1, …, DIM-1, …}
  bool isSdSchemeDefault() const;
  
  /// returns an iterator pair representing all mapped grid points of the surroundig box;
  /// mapped points get Flags flagsToSet (default = Flag::NONE) set
  template<Depth D, typename Grid, typename Map>
  typename std::enable_if<
      D == 0,
      thrust::pair<
          thrust::transform_iterator<
              GridPointsFunctor<DIM, REAL, N, Grid, Map>,
              thrust::zip_iterator<
                  thrust::tuple<
                      thrust::counting_iterator<NrPoints>,
                      thrust::constant_iterator<Box<N>>>>>,
          thrust::transform_iterator<
              GridPointsFunctor<DIM, REAL, N, Grid, Map>,
              thrust::zip_iterator<
                  thrust::tuple<
                      thrust::counting_iterator<NrPoints>,
                      thrust::constant_iterator<Box<N>>>>>>>::type
  makePointIteratorPair(Grid grid, Map map, Flags flagsToSet = Flag::NONE);
  
  /// actually the same like makePointIteratorPair<0> but without Depth as template parameter
  template<typename Grid, typename Map>
  thrust::pair<
      thrust::transform_iterator<
          GridPointsFunctor<DIM, REAL, N, Grid, Map>,
          thrust::zip_iterator<
              thrust::tuple<
                  thrust::counting_iterator<NrPoints>,
                  thrust::constant_iterator<Box<N>>>>>,
      thrust::transform_iterator<
          GridPointsFunctor<DIM, REAL, N, Grid, Map>,
          thrust::zip_iterator<
              thrust::tuple<
                  thrust::counting_iterator<NrPoints>,
                  thrust::constant_iterator<Box<N>>>>>>
  makePointIteratorPairOverRoot(Grid grid, Map map, Flags flagsToSet = Flag::NONE);
  
  /// returns an iterator pair representing all mapped grid points of all leaves;
  /// mapped points get Flags flagsToSet (default = Flag::NONE) set
  template<Depth D, typename Grid, typename Map>
  typename std::enable_if<
      D == -1,
      thrust::pair<
          thrust::transform_iterator<
              GridPointsFunctor<DIM, REAL, N, Grid, Map>,
              thrust::zip_iterator<
                  thrust::tuple<
                      thrust::counting_iterator<NrPoints>,
                      thrust::permutation_iterator<
                          typename ThrustSystem<A>::ConstBoxIterator<N>,
                          thrust::transform_iterator<IDivideByFunctor, thrust::counting_iterator<NrPoints>>>>>>,
          thrust::transform_iterator<
              GridPointsFunctor<DIM, REAL, N, Grid, Map>,
              thrust::zip_iterator<
                  thrust::tuple<
                      thrust::counting_iterator<NrPoints>,
                      thrust::permutation_iterator<
                          typename ThrustSystem<A>::ConstBoxIterator<N>,
                          thrust::transform_iterator<IDivideByFunctor, thrust::counting_iterator<NrPoints>>>>>>>>::type
  makePointIteratorPair(Grid grid, Map map, Flags flagsToSet = Flag::NONE);
  
  /// actually the same like makePointIteratorPair<-1> but without Depth as template parameter
  template<typename Grid, typename Map>
  thrust::pair<
      thrust::transform_iterator<
          GridPointsFunctor<DIM, REAL, N, Grid, Map>,
          thrust::zip_iterator<
              thrust::tuple<
                  thrust::counting_iterator<NrPoints>,
                  thrust::permutation_iterator<
                      typename ThrustSystem<A>::ConstBoxIterator<N>,
                      thrust::transform_iterator<IDivideByFunctor, thrust::counting_iterator<NrPoints>>>>>>,
      thrust::transform_iterator<
          GridPointsFunctor<DIM, REAL, N, Grid, Map>,
          thrust::zip_iterator<
              thrust::tuple<
                  thrust::counting_iterator<NrPoints>,
                  thrust::permutation_iterator<
                      typename ThrustSystem<A>::ConstBoxIterator<N>,
                      thrust::transform_iterator<IDivideByFunctor, thrust::counting_iterator<NrPoints>>>>>>>
  makePointIteratorPairOverLeaves(Grid grid, Map map, Flags flagsToSet = Flag::NONE);
  
  /// returns an iterator pair representing all mapped grid points of all boxes in Depth depth;
  /// mapped points get Flags flagsToSet (default = Flag::NONE) set
  template<Depth D, typename Grid, typename Map>
  typename std::enable_if<
      D != 0 && D != -1,
      thrust::pair<
          thrust::transform_iterator<
              GridPointsFunctor<DIM, REAL, N, Grid, Map>,
              thrust::zip_iterator<
                  thrust::tuple<
                      thrust::counting_iterator<NrPoints>,
                      thrust::transform_iterator<
                          DecrementDepthToFunctor<N>,
                          thrust::permutation_iterator<
                              typename ThrustSystem<A>::ConstBoxIterator<N>,
                              thrust::permutation_iterator<
                                  typename ThrustSystem<A>::Vector<NrBoxes>::const_iterator,
                                  thrust::transform_iterator<
                                      IDivideByFunctor,
                                      thrust::counting_iterator<NrPoints>>>>>>>>,
          thrust::transform_iterator<
              GridPointsFunctor<DIM, REAL, N, Grid, Map>,
              thrust::zip_iterator<
                  thrust::tuple<
                      thrust::counting_iterator<NrPoints>,
                      thrust::transform_iterator<
                          DecrementDepthToFunctor<N>,
                          thrust::permutation_iterator<
                              typename ThrustSystem<A>::ConstBoxIterator<N>,
                              thrust::permutation_iterator<
                                  typename ThrustSystem<A>::Vector<NrBoxes>::const_iterator,
                                  thrust::transform_iterator<
                                      IDivideByFunctor,
                                      thrust::counting_iterator<NrPoints>>>>>>>>>>::type
  makePointIteratorPair(Grid grid, Map map, Flags flagsToSet = Flag::NONE);
  
  /// actually the same like makePointIteratorPair<N> (N!=0 && N!=-1) 
  /// but with Depth as normal input parameter instead of template parameter
  template<typename Grid, typename Map>
  thrust::pair<
      thrust::transform_iterator<
          GridPointsFunctor<DIM, REAL, N, Grid, Map>,
          thrust::zip_iterator<
              thrust::tuple<
                  thrust::counting_iterator<NrPoints>,
                  thrust::transform_iterator<
                      DecrementDepthToFunctor<N>,
                      thrust::permutation_iterator<
                          typename ThrustSystem<A>::ConstBoxIterator<N>,
                          thrust::permutation_iterator<
                              typename ThrustSystem<A>::Vector<NrBoxes>::const_iterator,
                              thrust::transform_iterator<
                                  IDivideByFunctor,
                                  thrust::counting_iterator<NrPoints>>>>>>>>,
      thrust::transform_iterator<
          GridPointsFunctor<DIM, REAL, N, Grid, Map>,
          thrust::zip_iterator<
              thrust::tuple<
                  thrust::counting_iterator<NrPoints>,
                  thrust::transform_iterator<
                      DecrementDepthToFunctor<N>,
                      thrust::permutation_iterator<
                          typename ThrustSystem<A>::ConstBoxIterator<N>,
                          thrust::permutation_iterator<
                              typename ThrustSystem<A>::Vector<NrBoxes>::const_iterator,
                              thrust::transform_iterator<
                                  IDivideByFunctor,
                                  thrust::counting_iterator<NrPoints>>>>>>>>>
  makePointIteratorPairOverDepth(Depth depth, Grid grid, Map map, Flags flagsToSet = Flag::NONE);
  
  /// returns an iterator pair representing all mapped grid points of all boxes in Depth depth that
  /// have flagsToCheck set
  template<typename Grid, typename Map>
  thrust::pair<
      thrust::transform_iterator<
          GridPointsFunctor<DIM, REAL, N, Grid, Map>,
          thrust::zip_iterator<
              thrust::tuple<
                  thrust::counting_iterator<NrPoints>,
                  thrust::transform_iterator<
                      DecrementDepthToFunctor<N>,
                      thrust::permutation_iterator<
                          typename ThrustSystem<A>::ConstBoxIterator<N>,
                          thrust::permutation_iterator<
                              typename ThrustSystem<A>::Vector<NrBoxes>::const_iterator,
                              thrust::transform_iterator<
                                  IDivideByFunctor,
                                  thrust::counting_iterator<NrPoints>>>>>>>>,
      thrust::transform_iterator<
          GridPointsFunctor<DIM, REAL, N, Grid, Map>,
          thrust::zip_iterator<
              thrust::tuple<
                  thrust::counting_iterator<NrPoints>,
                  thrust::transform_iterator<
                      DecrementDepthToFunctor<N>,
                      thrust::permutation_iterator<
                          typename ThrustSystem<A>::ConstBoxIterator<N>,
                          thrust::permutation_iterator<
                              typename ThrustSystem<A>::Vector<NrBoxes>::const_iterator,
                              thrust::transform_iterator<
                                  IDivideByFunctor,
                                  thrust::counting_iterator<NrPoints>>>>>>>>>
  makePointIteratorPairByFlags(Depth depth, Flags flagsToCheck, Grid grid, Map map, Flags flagsToSet = Flag::NONE);
  
  /// removes all leaves (and corresponding ancestors) that have the Flags flags NOT set;
  /// returns the number of removed leaves
  NrBoxes remove(Flags flags);
  
  /// searches for the points given by the range [points.first, points.second) on Depth depth (default = -1);
  /// the resulting range [res_begin, res_begin + (points.second - points.first)) will contain the corresponding
  /// search index or getInvalidNrBoxes() if the point/box is not contained in the ImplicitBoxTree
  template<typename BoxIteratorPair, typename NrBoxesIterator>
  void searchByIterators(BoxIteratorPair points, Depth depth, NrBoxesIterator res_begin);
  
  /// same as void searchByIterators(…), but returns a vector containing the resulting range
  template<typename BoxIteratorPair>
  typename ThrustSystem<A>::Vector<NrBoxes> searchByIterators(BoxIteratorPair points, Depth depth = getLeafDepth());
  
  /// same as void searchByIterators(…), but searches for the points
  /// given by the coordinates {points[0], …, points[DIM-1] ; points[DIM], …, points[2*DIM-1] ; …}
  template<typename NrBoxesIterator>
  void searchByPoints(const typename ThrustSystem<A>::Vector<REAL>& points, Depth depth, NrBoxesIterator res_begin);
  
  /// same as void searchByPoints(…), but returns a vector containing the resulting range
  typename ThrustSystem<A>::Vector<NrBoxes> searchByPoints(const typename ThrustSystem<A>::Vector<REAL>& points,
                                                           Depth depth = getLeafDepth());
  
  /// searches for the points given by the range [points.first, points.second) on Depth depth (default = -1);
  /// returns a sorted, unique vector whose entries are the indices of the boxes in Depth depth
  /// that are hit by the points
  template<typename BoxIteratorPair>
  typename ThrustSystem<A>::Vector<NrBoxes> searchCompacted(BoxIteratorPair points, Depth depth = getLeafDepth());
  
  /// sets Flags flags in all boxes of Depth depth;
  /// returns the number of boxes in Depth depth that had Flags flags not set before
  NrBoxes setFlags(Flags flags, Depth depth = getLeafDepth());
  
  /// sets Flags flags in the boxes on Depth depth that contain the points 
  /// given by the range [points.first, points.second);
  /// returns the number of boxes in Depth depth that had Flags flags not set before
  template<typename BoxIteratorPair>
  NrBoxes setFlagsByIterators(BoxIteratorPair points, Flags flags, Depth depth = getLeafDepth());
  
  /// sets Flags flags in the boxes on Depth depth that contain the points
  /// given by the coordinates {points[0], …, points[DIM-1] ; points[DIM], …, points[2*DIM-1] ; …};
  /// returns the number of boxes in Depth depth that had Flags flags not set before
  NrBoxes setFlagsByPoints(const typename ThrustSystem<A>::Vector<REAL>& points,
                           Flags flags, Depth depth = getLeafDepth());
  
  /// sets Flags flags in the boxes whose indices on Depth depth are contained in sorted and unique searchVector;
  /// returns the number of boxes in Depth depth that had Flags flags not set before
  NrBoxes setFlagsBySearch(const typename ThrustSystem<A>::Vector<NrBoxes>& searchVector,
                           Flags flags, Depth depth = getLeafDepth());
  
  /// sets Flags flags in the i-th box on Depth depth if stencil_begin[i]==true;
  /// returns the number of boxes in Depth depth that had Flags flags not set before
  template<typename BoolIterator>
  NrBoxes setFlagsByStencil(BoolIterator stencil_begin, Flags flags, Depth depth = getLeafDepth());
  
  /// subdivides all leaves that have the Flags flags set;
  /// returns the number of subdivided boxes;
  NrBoxes subdivide(Flags flags = Flag::SD);
  
  /// computes the right stochastic (row-stochastic) transition matrix of Depth depth;
  /// the resulting matrix is stored in res;
  /// if useAbsoluteValues, the integer number of points are computed, else the actual definition,
  /// i.e. REAL(nMappedPointsInTargetBox) / REAL(nGridPointsPerBox);
  /// if useRowMajor, the matrix entries are sorted in row-major order, else in column-major order;
  /// if useInvalidNrBoxes, the matrix also contains the values for not hit boxes in column (this->count(depth) + 1)
  template<typename T, typename BoxIteratorPair>
  void transitionMatrix(BoxIteratorPair points,
                        Depth depth, NrPoints nPointsPerBox,
                        bool useAbsoluteValues, bool useRowMajor, bool useInvalidNrBoxes,
                        CooMatrix<A, T>& res);
  
  /// same as void transition_matrix(…) but the matrix is returned;
  /// Grid grid and Map map are used to compute the matrix for the surroundig box
  template<typename T, typename Grid, typename Map>
  CooMatrix<A, T> transitionMatrixForRoot(Grid grid, Map map,
                                          bool useAbsoluteValues = false,
                                          bool useRowMajor = false,
                                          bool useInvalidNrBoxes = false);
  
  /// same as void transition_matrix(…) but the matrix is returned;
  /// Grid grid and Map map are used to compute the matrix for all leaves
  template<typename T, typename Grid, typename Map>
  CooMatrix<A, T> transitionMatrixForLeaves(Grid grid, Map map,
                                            bool useAbsoluteValues = false, 
                                            bool useRowMajor = false,
                                            bool useInvalidNrBoxes = false);
  
  /// same as void transition_matrix(…) but the matrix is returned;
  /// Grid grid and Map map are used to compute the matrix for the boxes on Depth depth
  template<typename T, typename Grid, typename Map>
  CooMatrix<A, T> transitionMatrixForDepth(Depth depth, Grid grid, Map map,
                                           bool useAbsoluteValues = false,
                                           bool useRowMajor = false,
                                           bool useInvalidNrBoxes = false);
  
  /// see setFlags, but unsets all Flags flags;
  /// returns the number of boxes in which at least one Flag of flags is unset
  NrBoxes unsetFlags(Flags flags, Depth depth = getLeafDepth());
  
  template<typename BoxIteratorPair>
  NrBoxes unsetFlagsByIterators(BoxIteratorPair points, Flags flags, Depth depth = getLeafDepth());
  
  NrBoxes unsetFlagsByPoints(const typename ThrustSystem<A>::Vector<REAL>& points,
                             Flags flags, Depth depth = getLeafDepth());
  
  NrBoxes unsetFlagsBySearch(const typename ThrustSystem<A>::Vector<NrBoxes>& searchVector,
                             Flags flags, Depth depth = getLeafDepth());
  
  template<typename BoolIterator>
  NrBoxes unsetFlagsByStencil(BoolIterator stencil_begin, Flags flags, Depth depth = getLeafDepth());
  
  /// unsubdivides all leaves whose parent has the Flags flags set;
  /// returns the number of unsubdivided boxes
  NrBoxes unsubdivide(Flags flags = Flag::SD);
  
  
  /// reference functions ---------------------------------------------------------------------------------------------
  
  const typename ThrustSystem<A>::Vector<REAL>& getCenter() const;
  template<typename REAL2>
  typename ThrustSystem<A>::Vector<REAL>& setCenter(const typename ThrustSystem<A>::Vector<REAL2>& center);
  
  const typename ThrustSystem<A>::Vector<REAL>& getRadius() const;
  template<typename REAL2>
  typename ThrustSystem<A>::Vector<REAL>& setRadius(const typename ThrustSystem<A>::Vector<REAL2>& radius);
  
  const typename ThrustSystem<A>::Vector<Dimension>& getSdScheme() const;
  template<typename UInt>
  typename ThrustSystem<A>::Vector<Dimension>& setSdScheme(const typename ThrustSystem<A>::Vector<UInt>& sdScheme);
  
  const typename ThrustSystem<A>::Vector<Depth>& getSdCount() const;
  
  const typename ThrustSystem<A>::Vector<Depth>& getDepthVector() const;
  
  const typename ThrustSystem<A>::Vector<Flags>& getFlagsVector() const;
  
  uint64_t getNHostBufferBytes() const;
  uint64_t setNHostBufferBytes(uint64_t nHostBufferBytes);
  
  
  /// shorthands for special BoxIterators
  
  typename ThrustSystem<A>::ConstBoxIterator<N> begin() const;
  typename ThrustSystem<A>::BoxIterator<N> begin();
  typename ThrustSystem<A>::ConstBoxIterator<N> end() const;
  typename ThrustSystem<A>::BoxIterator<N> end();
  thrust::reverse_iterator<typename ThrustSystem<A>::ConstBoxIterator<N>> rbegin() const;
  thrust::reverse_iterator<typename ThrustSystem<A>::BoxIterator<N>> rbegin();
  thrust::reverse_iterator<typename ThrustSystem<A>::ConstBoxIterator<N>> rend() const;
  thrust::reverse_iterator<typename ThrustSystem<A>::BoxIterator<N>> rend();
  
  /// outstream operator
  template<Architecture A2, Dimension DIM2, typename REAL2, Depth N2>
  friend std::ostream& operator<<(std::ostream& os, const ImplicitBoxTree<A2, DIM2, REAL2, N2>& implicitBoxTree);
  
  
  /// some class (memory) functions -----------------------------------------------------------------------------------
  
  /// returns the dimension of the phase space; same as DIM
  Dimension dim() const;
  
  /// allocates enough memory for the "Box vectors" but does not change their size
  void reserve(NrBoxes n);
  
  /// resizes each "Box vector" to n
  void resize(NrBoxes n);
  
  /// shrinks the capacity of the "Box vectors" to exactly fit their elements
  void shrinkToFit();
  
  /// returns the number of stored leaves
  NrBoxes size() const;
  
  
 private:
  // center coordinates of the surroundig box
  typename ThrustSystem<A>::Vector<REAL> center_;
  // radii of the surroundig box
  typename ThrustSystem<A>::Vector<REAL> radius_;
  
  // subdivision scheme,
  // i.e. sdScheme_[depth] gives the coordinate direction in which a box of Depth depth would be subdivided
  typename ThrustSystem<A>::Vector<Dimension> sdScheme_;
  // sdCount_[depth*DIM + dim] gives the number of bisections in coordinate direction dim until Depth depth,
  // i.e. kind of exclusive_scan of sdScheme_
  typename ThrustSystem<A>::Vector<Depth> sdCount_;
  
  // vector storing the bit pattern for each box
  typename ThrustSystem<A>::Vector<BitPattern<N>> bitPatternVector_;
  // vector storing the Depth for each box
  typename ThrustSystem<A>::Vector<Depth> depthVector_;
  // vector storing the Flags for each box
  typename ThrustSystem<A>::Vector<Flags> flagsVector_;
  
  // vector storing the index for each box the ancestor box in Depth depthOfIndices_ has
  typename ThrustSystem<A>::Vector<NrBoxes> indicesInDepth_;
  // Depth corresponding to indicesInDepth_
  Depth depthOfIndices_;
  
  // vector containing the leaf indices where in the ImplicitBoxTree a "new" Box in Depth depth "starts" that has
  // Flags flagsToCheckForLeafIndices_ set
  typename ThrustSystem<A>::Vector<NrBoxes> leafIndicesFromDepth_;
  // Depth corresponding to leafIndicesFromDepth_
  Depth depthOfLeafIndices_;
  // Flags corresponding to leafIndicesFromDepth_
  Flags flagsToCheckForLeafIndices_;
  
  uint64_t nHostBufferBytes_; // number of bytes that host algorithms can temporarily allocate
};

} // namespace b12


#ifdef INCLUDE_IMPLICIT_BOX_TREE_MEMBER_DEFINITIONS
#include "ImplicitBoxTree.hpp"
#endif
