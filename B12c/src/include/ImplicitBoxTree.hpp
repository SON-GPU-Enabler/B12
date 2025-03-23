
#include <fstream>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/logical.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/replace.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/unique.h>


namespace b12 {

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline ImplicitBoxTree<A, DIM, REAL, N>::ImplicitBoxTree()
    : center_(DIM, 0.5), radius_(DIM, 0.5),
      sdScheme_(thrust::make_transform_iterator(thrust::make_counting_iterator(Depth(0)),
                                                thrust::placeholders::_1 % DIM),
                thrust::make_transform_iterator(thrust::make_counting_iterator(getMaxDepth<N>()),
                                                thrust::placeholders::_1 % DIM)),
      sdCount_(),
      bitPatternVector_(), depthVector_(), flagsVector_(),
      indicesInDepth_(), depthOfIndices_(),
      leafIndicesFromDepth_(), depthOfLeafIndices_(), flagsToCheckForLeafIndices_(),
      nHostBufferBytes_(uint64_t(4) << 30)
{
  this->adaptSdCount();
  
  this->insertSurroundingBoxIfEmpty();
  
  ThrustSystem<A>::Memory::printDevice();
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<Architecture A2, Dimension DIM2, typename REAL2, Depth N2>
inline ImplicitBoxTree<A, DIM, REAL, N>::ImplicitBoxTree(const ImplicitBoxTree<A2, DIM2, REAL2, N2>& implicitBoxTree)
    : ImplicitBoxTree()
{
  this->setCenter<REAL2>(implicitBoxTree.center_);
  this->setRadius<REAL2>(implicitBoxTree.radius_);
  this->setSdScheme<Dimension>(implicitBoxTree.sdScheme_);
  this->adaptSdCount();
  
  bitPatternVector_.resize(implicitBoxTree.size());
  depthVector_.resize(implicitBoxTree.size());
  flagsVector_.resize(implicitBoxTree.size());
  
  indicesInDepth_.assign(implicitBoxTree.indicesInDepth_.begin(), implicitBoxTree.indicesInDepth_.end());
  depthOfIndices_ = implicitBoxTree.depthOfIndices_;
  leafIndicesFromDepth_.assign(implicitBoxTree.leafIndicesFromDepth_.begin(),
                               implicitBoxTree.leafIndicesFromDepth_.end());
  depthOfLeafIndices_ = implicitBoxTree.depthOfLeafIndices_;
  flagsToCheckForLeafIndices_ = implicitBoxTree.flagsToCheckForLeafIndices_;
  
  nHostBufferBytes_ = implicitBoxTree.nHostBufferBytes_;
  
  if (implicitBoxTree.depth() > N) {
    std::cout << "Warning: The maximal depth of the given ImplicitBoxTree exceeds new maximal depth. "
              << "Some boxes might be unsubdivided until new maximal depth is reached."
              << std::endl;
  }
  
  if (A == A2) {
    // decrement the Depth of implicitBoxTree's N2-boxes to N if their Depth is > N
    thrust::transform(typename ThrustSystem<A>::execution_policy(),
                      implicitBoxTree.begin(), implicitBoxTree.end(),
                      this->begin(),
                      DecrementDepthToFunctor<N2>(N));
  } else {
    typename ThrustSystem<A>::Vector<BitPattern<N2>> _bitPatternVector(implicitBoxTree.bitPatternVector_.begin(),
                                                                       implicitBoxTree.bitPatternVector_.end());
    typename ThrustSystem<A>::Vector<Depth> _depthVector(implicitBoxTree.depthVector_.begin(),
                                                         implicitBoxTree.depthVector_.end());
    typename ThrustSystem<A>::Vector<Flags> _flagsVector(implicitBoxTree.flagsVector_.begin(),
                                                         implicitBoxTree.flagsVector_.end());
    
    auto _begin = thrust::make_zip_iterator(thrust::make_tuple(_bitPatternVector.begin(),
                                                               _depthVector.begin(),
                                                               _flagsVector.begin()));
    
    // decrement the Depth of implicitBoxTree's N2-boxes to N if their Depth is > N
    thrust::transform(typename ThrustSystem<A>::execution_policy(),
                      _begin, _begin + implicitBoxTree.size(),
                      this->begin(),
                      DecrementDepthToFunctor<N2>(N));
  }
  
  // combine (and) flags of "equal" boxes
  thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(),
                                this->rbegin(), this->rend(), // keys
                                flagsVector_.rbegin(), // Note all reverse_iterators!
                                flagsVector_.rbegin(), // in-place trafo
                                IsSameBoxFunctor<N>(), // "segment operator" for keys
                                FlagsAndFunctor()); // first box of segment gets intersection of all flags
  
  // delete all of equal consecutive boxes but the first one
  // substract those boxes that had a sibling to prevent double counting
  auto box_new_end = thrust::unique(typename ThrustSystem<A>::execution_policy(),
                                    this->begin(), this->end(),
                                    IsSameBoxFunctor<N>());
  
  this->resize(box_new_end - this->begin());
  
  this->insertSurroundingBoxIfEmpty();
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename REAL2>
inline ImplicitBoxTree<A, DIM, REAL, N>::ImplicitBoxTree(const typename ThrustSystem<A>::Vector<REAL2>& center,
                                                         const typename ThrustSystem<A>::Vector<REAL2>& radius)
    : ImplicitBoxTree()
{
  this->setCenter<REAL2>(center);
  this->setRadius<REAL2>(radius);
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename REAL2, typename UInt>
inline ImplicitBoxTree<A, DIM, REAL, N>::ImplicitBoxTree(const typename ThrustSystem<A>::Vector<REAL2>& center,
                                                         const typename ThrustSystem<A>::Vector<REAL2>& radius,
                                                         const typename ThrustSystem<A>::Vector<UInt>& sdScheme)
    : ImplicitBoxTree()
{
  this->setCenter<REAL2>(center);
  this->setRadius<REAL2>(radius);
  this->setSdScheme<UInt>(sdScheme);
  this->adaptSdCount();
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline ImplicitBoxTree<A, DIM, REAL, N>::ImplicitBoxTree(const std::string& fileName)
    : ImplicitBoxTree()
{
  bool isLegacyFile, isOpenable, wasSuccess;
  
  std::ifstream ifs;
  
  ifs.open(fileName, std::ifstream::in);
  if (ifs.good()) {
    isOpenable = true;
    // if fileName does not end with ".json", it is supposed to be a file written with B12a or B12b
    isLegacyFile = fileName.length() < 5 || fileName.compare(fileName.length() - 5, 5, ".json") != 0;
    if (isLegacyFile) {
      // reopen as binary
      ifs.close();
      ifs.open(fileName, std::ifstream::in | std::ifstream::binary);
      isOpenable = ifs.good();
    }
  } else {
    // try to open fileName+".json"
    ifs.close();
    ifs.open(fileName + ".json", std::ifstream::in);
    isOpenable = ifs.good();
    isLegacyFile = false;
  }
  
  if (isOpenable) {
    typename ThrustSystem<A>::Vector<BitPattern<MaxNBits::value>> _bitPatternVector;
    typename ThrustSystem<A>::Vector<Depth> _depthVector;
    typename ThrustSystem<A>::Vector<Flags> _flagsVector;
    
    if (isLegacyFile) {
      // read version numbers of B12
      uint32_t B12_VERSION_MAJOR_temp, B12_VERSION_MINOR_temp;
      if (ifs.good()) {
        ifs.read(reinterpret_cast<char*>(&B12_VERSION_MAJOR_temp), sizeof(uint32_t));
      }
      if (ifs.good()) {
        ifs.read(reinterpret_cast<char*>(&B12_VERSION_MINOR_temp), sizeof(uint32_t));
      }
      
      // read the "small" vectors: vector.size(), vector.data()
      uint64_t dim;
      if (ifs.good()) {
        ifs.read(reinterpret_cast<char*>(&dim), sizeof(uint64_t));
      }
      
      if (ifs.good()) {
        typename ThrustSystem<HOST>::Vector<double> _center(dim);
        ifs.read(reinterpret_cast<char*>(thrust::raw_pointer_cast(_center.data())), int(dim) * sizeof(double));
        this->setCenter<double>(_center);
      }
      
      if (ifs.good()) {
        typename ThrustSystem<HOST>::Vector<double> _radius(dim);
        ifs.read(reinterpret_cast<char*>(thrust::raw_pointer_cast(_radius.data())), int(dim) * sizeof(double));
        this->setRadius<double>(_radius);
      }
      
      if (ifs.good()) {
        // error in older versions: template argument N must be the same as it was stored !!!
        typename ThrustSystem<HOST>::Vector<Dimension> _sdScheme(int(getMaxDepth<N>()) + 1);
        ifs.read(reinterpret_cast<char*>(thrust::raw_pointer_cast(_sdScheme.data())),
                 (int(getMaxDepth<N>()) + 1) * sizeof(Dimension));
        this->setSdScheme<Dimension>(_sdScheme);
        this->adaptSdCount();
      }
      
      // read "box" vectors: vector.size(), vector.data()
      uint64_t vectorSize;
      if (ifs.good()) {
        ifs.read(reinterpret_cast<char*>(&vectorSize), sizeof(uint64_t));
      }
      if (ifs.good()) {
        typename ThrustSystem<HOST>::Vector<BitPattern<128>> _bitPatternVector_temp(vectorSize);
        ifs.read(reinterpret_cast<char*>(thrust::raw_pointer_cast(_bitPatternVector_temp.data())),
                 vectorSize * sizeof(BitPattern<128>));
        _bitPatternVector.assign(_bitPatternVector_temp.begin(), _bitPatternVector_temp.end());
      }
      
      if (ThrustSystem<A>::Memory::isOnHost()) {
        _depthVector.resize(vectorSize);
        if (ifs.good()) {
          ifs.read(reinterpret_cast<char*>(thrust::raw_pointer_cast(_depthVector.data())),
                   vectorSize * sizeof(Depth));
        }
        
        _flagsVector.resize(vectorSize);
        if (ifs.good()) {
          ifs.read(reinterpret_cast<char*>(thrust::raw_pointer_cast(_flagsVector.data())),
                   vectorSize * sizeof(Flags));
        }
      } else {
        if (ifs.good()) {
          typename ThrustSystem<HOST>::Vector<Depth> _depthVector_temp(vectorSize);
          ifs.read(reinterpret_cast<char*>(thrust::raw_pointer_cast(_depthVector_temp.data())),
                   vectorSize * sizeof(Depth));
          _depthVector.assign(_depthVector_temp.begin(), _depthVector_temp.end());
        }
        
        if (ifs.good()) {
          typename ThrustSystem<HOST>::Vector<Flags> _flagsVector_temp(vectorSize);
          ifs.read(reinterpret_cast<char*>(thrust::raw_pointer_cast(_flagsVector_temp.data())),
                                            vectorSize * sizeof(Flags));
          _flagsVector.assign(_flagsVector_temp.begin(), _flagsVector_temp.end());
        }
      }
      
      if (ifs.good()) {
        ifs.read(reinterpret_cast<char*>(&nHostBufferBytes_), sizeof(uint64_t));
      }
      
      wasSuccess = ifs.good();
      ifs.close();
    } else {
      std::string fileString;
      if (ifs.good()) {
        std::stringstream strStream;
        strStream << ifs.rdbuf();
        fileString = strStream.str();
      }
      wasSuccess = ifs.good();
      ifs.close();
      
      {
        typename ThrustSystem<A>::Vector<REAL> _center(getNumberVectorFromJSONString<REAL>(fileString, "center", DIM));
        this->setCenter<REAL>(_center);
      }
      
      {
        typename ThrustSystem<A>::Vector<REAL> _radius(getNumberVectorFromJSONString<REAL>(fileString, "radius", DIM));
        this->setRadius<REAL>(_radius);
      }
      
      {
        typename ThrustSystem<A>::Vector<Dimension> _sdScheme(getNumberVectorFromJSONString<Dimension>(fileString,
                                                                                                       "sdScheme",
                                                                                                       129));
        this->setSdScheme<Dimension>(_sdScheme);
        this->adaptSdCount();
      }
      
      uint64_t nLeaves = getNumberFromJSONString<uint64_t>(fileString, "nLeaves");
      
      {
        auto _bitPatternVector_temp = getNumberVectorFromJSONString<BitPattern<MaxNBits::value>>(fileString,
                                                                                                 "bitPatternVector",
                                                                                                 nLeaves);
        _bitPatternVector.assign(_bitPatternVector_temp.begin(), _bitPatternVector_temp.end());
      }
      
      {
        auto _depthVector_temp = getNumberVectorFromJSONString<Depth>(fileString, "depthVector", nLeaves);
        _depthVector.assign(_depthVector_temp.begin(), _depthVector_temp.end());
      }
      
      {
        auto _flagsVector_temp = getNumberVectorFromJSONString<Flags>(fileString, "flagsVector", nLeaves);
        _flagsVector.assign(_flagsVector_temp.begin(), _flagsVector_temp.end());
      }
      
      nHostBufferBytes_ = getNumberFromJSONString<uint64_t>(fileString, "nHostBufferBytes");
    }
    
    if (wasSuccess) {
      Depth maximalDepth = thrust::reduce(typename ThrustSystem<A>::execution_policy(),
                                          _depthVector.begin(), _depthVector.end(),
                                          Depth(0),
                                          thrust::maximum<Depth>());
      
      if (maximalDepth > MaxNBits::value) {
        std::cout << "Warning: The maximal depth read from the given file (=" << maximalDepth 
                  << ") exceeds the software's available maximum (=" << MaxNBits::value << "). "
                  << "This might result in undefined behaviour."
                  << std::endl;
      } else if (maximalDepth > N) {
        std::cout << "Warning: The maximal depth read from the given file (=" << maximalDepth 
                  << ") exceeds new maximal depth (=" << N << "). "
                  << "Some boxes might be unsubdivided until new maximal depth is reached."
                  << std::endl;
      }
      
      this->resize(_bitPatternVector.size());
      
      auto _begin = thrust::make_zip_iterator(thrust::make_tuple(_bitPatternVector.begin(),
                                                                 _depthVector.begin(),
                                                                 _flagsVector.begin()));
      
      // decrement the Depth of read boxes to N if their Depth is > N
      thrust::transform(typename ThrustSystem<A>::execution_policy(),
                        _begin, _begin + _bitPatternVector.size(),
                        this->begin(),
                        DecrementDepthToFunctor<MaxNBits::value>(N));
      
      // combine (and) flags of "equal" boxes
      thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(),
                                    this->rbegin(), this->rend(), // keys
                                    flagsVector_.rbegin(), // Note all reverse_iterators!
                                    flagsVector_.rbegin(), // in-place trafo
                                    IsSameBoxFunctor<N>(), // "segment operator" for keys
                                    FlagsAndFunctor()); // first box of segment gets intersection of all flags
      
      // delete all of equal consecutive boxes but the first one
      // substract those boxes that had a sibling to prevent double counting
      auto box_new_end = thrust::unique(typename ThrustSystem<A>::execution_policy(),
                                        this->begin(), this->end(),
                                        IsSameBoxFunctor<N>());
      
      this->resize(box_new_end - this->begin());
    } else {
      std::cout << "Warning: An error occurred during reading." << std::endl;
    }
  } else {
    std::cout << "Warning: File could not be opened." << std::endl;
  }
}
  
template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<Architecture A2, Dimension DIM2, typename REAL2, Depth N2>
inline ImplicitBoxTree<A, DIM, REAL, N>& ImplicitBoxTree<A, DIM, REAL, N>::operator=(
    const ImplicitBoxTree<A2, DIM2, REAL2, N2>& implicitBoxTree)
{
  this->setCenter<REAL2>(implicitBoxTree.center_);
  this->setRadius<REAL2>(implicitBoxTree.radius_);
  this->setSdScheme<Dimension>(implicitBoxTree.sdScheme_);
  this->adaptSdCount();
  
  bitPatternVector_.resize(implicitBoxTree.size());
  depthVector_.resize(implicitBoxTree.size());
  flagsVector_.resize(implicitBoxTree.size());
  
  indicesInDepth_.assign(implicitBoxTree.indicesInDepth_.begin(), implicitBoxTree.indicesInDepth_.end());
  depthOfIndices_ = implicitBoxTree.depthOfIndices_;
  leafIndicesFromDepth_.assign(implicitBoxTree.leafIndicesFromDepth_.begin(),
                               implicitBoxTree.leafIndicesFromDepth_.end());
  depthOfLeafIndices_ = implicitBoxTree.depthOfLeafIndices_;
  flagsToCheckForLeafIndices_ = implicitBoxTree.flagsToCheckForLeafIndices_;
  
  nHostBufferBytes_ = implicitBoxTree.nHostBufferBytes_;
  
  if (implicitBoxTree.depth() > N) {
    std::cout << "Warning: The maximal depth of the given ImplicitBoxTree exceeds new maximal depth. "
              << "Some boxes might be unsubdivided until new maximal depth is reached."
              << std::endl;
  }
  
  if (A == A2) {
    // decrement the Depth of implicitBoxTree's N2-boxes to N if their Depth is > N
    thrust::transform(typename ThrustSystem<A>::execution_policy(),
                      implicitBoxTree.begin(), implicitBoxTree.end(),
                      this->begin(),
                      DecrementDepthToFunctor<N2>(N));
  } else {
    typename ThrustSystem<A>::Vector<BitPattern<N2>> _bitPatternVector(implicitBoxTree.bitPatternVector_.begin(),
                                                                       implicitBoxTree.bitPatternVector_.end());
    typename ThrustSystem<A>::Vector<Depth> _depthVector(implicitBoxTree.depthVector_.begin(),
                                                         implicitBoxTree.depthVector_.end());
    typename ThrustSystem<A>::Vector<Flags> _flagsVector(implicitBoxTree.flagsVector_.begin(),
                                                         implicitBoxTree.flagsVector_.end());
    
    auto _begin = thrust::make_zip_iterator(thrust::make_tuple(_bitPatternVector.begin(),
                                                               _depthVector.begin(),
                                                               _flagsVector.begin()));
    
    // decrement the Depth of implicitBoxTree's N2-boxes to N if their Depth is > N
    thrust::transform(typename ThrustSystem<A>::execution_policy(),
                      _begin, _begin + implicitBoxTree.size(),
                      this->begin(),
                      DecrementDepthToFunctor<N2>(N));
  }
  
  // combine (and) flags of "equal" boxes
  thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(),
                                this->rbegin(), this->rend(), // keys
                                flagsVector_.rbegin(), // Note all reverse_iterators!
                                flagsVector_.rbegin(), // in-place trafo
                                IsSameBoxFunctor<N>(), // "segment operator" for keys
                                FlagsAndFunctor()); // first box of segment gets intersection of all flags
  
  // delete all of equal consecutive boxes but the first one
  // substract those boxes that had a sibling to prevent double counting
  auto box_new_end = thrust::unique(typename ThrustSystem<A>::execution_policy(),
                                    this->begin(), this->end(),
                                    IsSameBoxFunctor<N>());
  
  this->resize(box_new_end - this->begin());
  
  this->insertSurroundingBoxIfEmpty();
  
  return *this;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline void ImplicitBoxTree<A, DIM, REAL, N>::save(const std::string& fileName) const
{
  std::string fileNameExtension;
  if (fileName.length() < 5 || fileName.compare(fileName.length() - 5, 5, ".json") != 0) {
    fileNameExtension = ".json";
  }
  
  std::ofstream ofs(fileName + fileNameExtension, std::ofstream::out | std::ofstream::trunc);
  
  if (ofs.good()) {
    ofs << "{";
  } else {
    std::cout << "Warning: File could not be opened." << std::endl;
  }
  
  if (ofs.good()) {
    putNumberIntoJSONStream(ofs, "B12_VERSION_MAJOR", B12_VERSION_MAJOR);
    ofs << ",";
    putNumberIntoJSONStream(ofs, "B12_VERSION_MINOR", B12_VERSION_MINOR);
    ofs << ",";
  }
  
  if (ofs.good()) {
    std::vector<REAL> _center(center_.size());
    thrust::copy(center_.begin(), center_.end(), _center.begin());
    putNumberVectorIntoJSONStream(ofs, "center", _center);
    ofs << ",";
    _center.clear();
    _center.shrink_to_fit();
  }
  
  if (ofs.good()) {
    std::vector<REAL> _radius(radius_.size());
    thrust::copy(radius_.begin(), radius_.end(), _radius.begin());
    putNumberVectorIntoJSONStream(ofs, "radius", _radius);
    ofs << ",";
    _radius.clear();
    _radius.shrink_to_fit();
  }
  
  if (ofs.good()) {
    std::vector<Dimension> _sdScheme(sdScheme_.size());
    thrust::copy(sdScheme_.begin(), sdScheme_.end(), _sdScheme.begin());
    putNumberVectorIntoJSONStream(ofs, "sdScheme", _sdScheme);
    ofs << ",";
    _sdScheme.clear();
    _sdScheme.shrink_to_fit();
  }
  
  if (ofs.good()) {
    putNumberIntoJSONStream(ofs, "nLeaves", bitPatternVector_.size());
    ofs << ",";
  }
  
  if (ofs.good()) {
    std::vector<BitPattern<N>> _bitPatternVector(bitPatternVector_.size());
    thrust::copy(bitPatternVector_.begin(), bitPatternVector_.end(), _bitPatternVector.begin());
    putNumberVectorIntoJSONStream(ofs, "bitPatternVector", _bitPatternVector);
    ofs << ",";
    _bitPatternVector.clear();
    _bitPatternVector.shrink_to_fit();
  }
  
  if (ofs.good()) {
    std::vector<Depth> _depthVector(depthVector_.size());
    thrust::copy(depthVector_.begin(), depthVector_.end(), _depthVector.begin());
    putNumberVectorIntoJSONStream(ofs, "depthVector", _depthVector);
    ofs << ",";
    _depthVector.clear();
    _depthVector.shrink_to_fit();
  }
  
  if (ofs.good()) {
    std::vector<Flags> _flagsVector(flagsVector_.size());
    thrust::copy(flagsVector_.begin(), flagsVector_.end(), _flagsVector.begin());
    putNumberVectorIntoJSONStream(ofs, "flagsVector", _flagsVector);
    ofs << ",";
    _flagsVector.clear();
    _flagsVector.shrink_to_fit();
  }
  
  if (ofs.good()) {
    putNumberIntoJSONStream(ofs, "nHostBufferBytes", nHostBufferBytes_);
  }
  
  if (ofs.good()) {
    ofs << "\n}";
  }
  
  if (! ofs.good()) {
    std::cout << "Warning: An error occurred during writing." << std::endl;
  }
  
  ofs.close();
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline void ImplicitBoxTree<A, DIM, REAL, N>::saveForGaio(const std::string& fileName) const
{
  if (ThrustSystem<A>::Memory::isOnHost()) {
    // open a binary file, whose content - if any - is discarded
    std::ofstream file(fileName, std::ofstream::binary | std::ofstream::trunc);
    
    // store the "small" vectors
    int dim = DIM;
    file.write(reinterpret_cast<const char*>(&dim), sizeof(int));
    int size_of_coordinatetype = sizeof(REAL);
    file.write(reinterpret_cast<const char*>(&size_of_coordinatetype), sizeof(int));
    file.write(reinterpret_cast<const char*>(thrust::raw_pointer_cast(center_.data())), dim * sizeof(REAL));
    file.write(reinterpret_cast<const char*>(thrust::raw_pointer_cast(radius_.data())), dim * sizeof(REAL));
    for (int i = 0; i <= 500 * dim; ++i) {
      int sd_i = sdScheme_[i % sdScheme_.size()];
      file.write(reinterpret_cast<const char*>(&sd_i), sizeof(int));
    }
    
    // store center coordinates of boxes and the depth
    double nBoxes = this->size();
    file.write(reinterpret_cast<const char*>(&nBoxes), sizeof(double));
    REAL coords[2 * DIM + 1];
    ComputeCoordinatesOfBoxFunctor<DIM, REAL, N> fun(thrust::raw_pointer_cast(center_.data()),
                                                     thrust::raw_pointer_cast(radius_.data()),
                                                     thrust::raw_pointer_cast(sdScheme_.data()),
                                                     thrust::raw_pointer_cast(sdCount_.data()),
                                                     this->isSdSchemeDefault());
    for (auto bi = this->begin(); bi != this->end(); ++bi) {
      fun(thrust::make_tuple(*bi, &(coords[0])));
      file.write(reinterpret_cast<const char*>(coords), dim * sizeof(REAL));
      double depth = getDepth<N>(*bi);
      file.write(reinterpret_cast<const char*>(&depth), sizeof(double));
      double flags = getFlags<N>(*bi);
      file.write(reinterpret_cast<const char*>(&flags), sizeof(double));
    }
  } else {
    ImplicitBoxTree<HOST, DIM, REAL, N>(*this).saveForGaio(fileName);
  }
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline void ImplicitBoxTree<A, DIM, REAL, N>::adaptSdCount()
{
  sdCount_.resize((int(getMaxDepth<N>()) + 2) * int(DIM));
  
  thrust::fill(typename ThrustSystem<A>::execution_policy(), sdCount_.begin(), sdCount_.end(), Depth(0));
  
  for (int i = 0; i < sdScheme_.size(); ++i) {
    thrust::copy(typename ThrustSystem<A>::execution_policy(),
                 sdCount_.begin() + i * int(DIM), sdCount_.begin() + (i + 1) * int(DIM),
                 sdCount_.begin() + (i + 1) * int(DIM));
    ++sdCount_[(i + 1) * int(DIM) + int(sdScheme_[i])];
  }
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline bool ImplicitBoxTree<A, DIM, REAL, N>::areAllLeavesOnDepth(Depth depth) const
{
  if (depth == getLeafDepth()) {
    
    // always true since getLeafDepth() is the depth of leaves
    return true;
    
  } else if (depth == getUnsubdividableDepth()) {
    
    // always false since unsubdividable boxes cannot be leaves
    return false;
    
  } else {
    
    return thrust::all_of(typename ThrustSystem<A>::execution_policy(),
                          depthVector_.begin(), depthVector_.end(),
                          thrust::placeholders::_1 == depth);
    
  }
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline void ImplicitBoxTree<A, DIM, REAL, N>::boxes(Depth depth, REAL * res_begin) const
{
  if (depth == 0) {
    
    thrust::copy(center_.begin(), center_.end(), res_begin);
    thrust::copy(radius_.begin(), radius_.end(), res_begin + DIM);
    
    Flags flags = thrust::reduce(typename ThrustSystem<A>::execution_policy(),
                                 flagsVector_.begin(), flagsVector_.end(),
                                 Flags(Flag::ALL),
                                 FlagsAndFunctor());
    
    typename ThrustSystem<A>::Vector<REAL> _flagsVector(1, flags);
    thrust::copy(_flagsVector.begin(), _flagsVector.end(), res_begin + 2 * DIM);
    
  } else if (depth == getLeafDepth() || this->areAllLeavesOnDepth(depth)) {
    
    auto zi_begin =
        thrust::make_zip_iterator(
            thrust::make_tuple(
                this->begin(), // boxes
                thrust::make_transform_iterator( // res_begin[0], res_begin[2*DIM+1], res_begin[4*DIM+2], …
                    thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                                    thrust::placeholders::_1 * NrPoints(2 * DIM + 1)),
                    AdvanceRealPointerFunctor<REAL>(res_begin))));
    
    thrust::for_each(typename ThrustSystem<A>::execution_policy(),
                     zi_begin, zi_begin + this->count(getLeafDepth()), // input
                     ComputeCoordinatesOfBoxFunctor<DIM, REAL, N>(thrust::raw_pointer_cast(center_.data()),
                                                                  thrust::raw_pointer_cast(radius_.data()),
                                                                  thrust::raw_pointer_cast(sdScheme_.data()),
                                                                  thrust::raw_pointer_cast(sdCount_.data()),
                                                                  this->isSdSchemeDefault()));
    
  } else if (depth == getUnsubdividableDepth()) {
    
    NrBoxes nLeaves = this->size();
    
    typename ThrustSystem<A>::Vector<BitPattern<N>> _bitPatternVector(nLeaves);
    typename ThrustSystem<A>::Vector<Depth> _depthVector(nLeaves);
    typename ThrustSystem<A>::Vector<Flags> _flagsVector(nLeaves);
    
    auto _box_begin = thrust::make_zip_iterator(thrust::make_tuple(_bitPatternVector.begin(),
                                                                   _depthVector.begin(),
                                                                   _flagsVector.begin()));
    
    // only copy unsubdividable boxes
    auto _box_end = thrust::copy_if(typename ThrustSystem<A>::execution_policy(),
                                    thrust::make_transform_iterator(this->begin(), DecrementDepthByFunctor<N>(1)),
                                    thrust::make_transform_iterator(this->end(), DecrementDepthByFunctor<N>(1)),
                                    this->isUnsubdividable_begin(), // stencil
                                    _box_begin, // output
                                    thrust::identity<bool>()); // predicate for stencil
                                  
    NrBoxes nBoxes = _box_end - _box_begin;
    
    // free some memory for later thrust::inclusive_scan_by_key and thrust::unique
    _bitPatternVector.resize(nBoxes);
    _bitPatternVector.shrink_to_fit();
    _depthVector.resize(nBoxes);
    _depthVector.shrink_to_fit();
    _flagsVector.resize(nBoxes);
    _flagsVector.shrink_to_fit();
    
    // update iterators because of possible new allocation
    _box_begin = thrust::make_zip_iterator(thrust::make_tuple(_bitPatternVector.begin(),
                                                              _depthVector.begin(),
                                                              _flagsVector.begin()));
    _box_end = _box_begin + nBoxes;
    
    thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(), // Note: all reverse_iterators
                                  thrust::make_reverse_iterator(_box_end), // keys_begin
                                  thrust::make_reverse_iterator(_box_begin), // keys_end
                                  _flagsVector.rbegin(), // values_begin
                                  _flagsVector.rbegin(), // in-place trafo
                                  IsSameBoxFunctor<N>(), // "segment operator" for keys
                                  FlagsAndFunctor());
    
    _box_end = thrust::unique(typename ThrustSystem<A>::execution_policy(),
                              _box_begin, _box_end,
                              IsSameBoxFunctor<N>());
    
    auto zi_begin =
        thrust::make_zip_iterator(
            thrust::make_tuple(
                // iterator over temporary boxes
                _box_begin,
                // res_begin[0], res_begin[2*DIM+1], res_begin[4*DIM+2], …
                thrust::make_transform_iterator(
                    thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                                    thrust::placeholders::_1 * NrPoints(2 * DIM + 1)),
                    AdvanceRealPointerFunctor<REAL>(res_begin))));
    
    thrust::for_each(typename ThrustSystem<A>::execution_policy(),
                     zi_begin, zi_begin + (_box_end - _box_begin), // input
                     ComputeCoordinatesOfBoxFunctor<DIM, REAL, N>(thrust::raw_pointer_cast(center_.data()),
                                                                  thrust::raw_pointer_cast(radius_.data()),
                                                                  thrust::raw_pointer_cast(sdScheme_.data()),
                                                                  thrust::raw_pointer_cast(sdCount_.data()),
                                                                  this->isSdSchemeDefault()));
    
  } else if (depth <= this->depth()) {
    
    NrBoxes nLeaves = this->size();
    
    typename ThrustSystem<A>::Vector<BitPattern<N>> _bitPatternVector(nLeaves);
    typename ThrustSystem<A>::Vector<Depth> _depthVector(nLeaves);
    typename ThrustSystem<A>::Vector<Flags> _flagsVector(nLeaves);
    
    // "output-flags-vector" splitted in two vectors to guarantee no overwriting
    auto itPair = thrust::reduce_by_key(typename ThrustSystem<A>::execution_policy(),
                                        thrust::make_transform_iterator(
                                            this->begin(), DecrementDepthToFunctor<N>(depth)), // keys_begin
                                        thrust::make_transform_iterator(
                                            this->end(), DecrementDepthToFunctor<N>(depth)), // keys_end
                                        flagsVector_.begin(), // values to reduce
                                        thrust::make_zip_iterator(
                                            thrust::make_tuple(_bitPatternVector.begin(),
                                                               _depthVector.begin(),
                                                               thrust::make_discard_iterator())), // output for keys
                                        _flagsVector.begin(), // output for reduced flags by keys
                                        IsSameBoxFunctor<N>(), // "segment operator" for keys
                                        FlagsAndFunctor()); // reduction operation
    
    NrBoxes nBoxes = itPair.second - _flagsVector.begin();
    
    // free some memory for later thrust::stable_partition
    _bitPatternVector.resize(nBoxes);
    _bitPatternVector.shrink_to_fit();
    _depthVector.resize(nBoxes);
    _depthVector.shrink_to_fit();
    _flagsVector.resize(nBoxes);
    _flagsVector.shrink_to_fit();
    
    auto _box_begin = thrust::make_zip_iterator(thrust::make_tuple(_bitPatternVector.begin(),
                                                                   _depthVector.begin(),
                                                                   _flagsVector.begin()));
    
    // order such that boxes with Depth >= depth precede boxes with Depth < depth
    auto _box_end = thrust::stable_partition(typename ThrustSystem<A>::execution_policy(),
                                             _box_begin, _box_begin + nBoxes,
                                             HasAtLeastDepthFunctor<N>(depth)); // predicate for partitioning
    
    auto zi_begin =
        thrust::make_zip_iterator(
            thrust::make_tuple(
                // iterator over temporary boxes
                _box_begin,
                // res_begin[0], res_begin[2*DIM+1], res_begin[4*DIM+2], …
                thrust::make_transform_iterator(
                    thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                                    thrust::placeholders::_1 * NrPoints(2 * DIM + 1)),
                    AdvanceRealPointerFunctor<REAL>(res_begin))));
    
    thrust::for_each(typename ThrustSystem<A>::execution_policy(),
                     zi_begin, zi_begin + (_box_end - _box_begin), // input
                     ComputeCoordinatesOfBoxFunctor<DIM, REAL, N>(thrust::raw_pointer_cast(center_.data()),
                                                                  thrust::raw_pointer_cast(radius_.data()),
                                                                  thrust::raw_pointer_cast(sdScheme_.data()),
                                                                  thrust::raw_pointer_cast(sdCount_.data()),
                                                                  this->isSdSchemeDefault()));
  }
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline typename ThrustSystem<A>::Vector<REAL> ImplicitBoxTree<A, DIM, REAL, N>::boxes(Depth depth) const
{
  typename ThrustSystem<A>::Vector<REAL> res(this->count(depth) * NrPoints(2 * DIM + 1));
  
  this->boxes(depth, thrust::raw_pointer_cast(res.data()));
  
  return res;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline NrBoxes ImplicitBoxTree<A, DIM, REAL, N>::changeFlags(Flags flagsToUnset, Flags flagsToSet, Depth depth)
{
  return this->changeFlagsByStencil(thrust::make_constant_iterator(true), flagsToUnset, flagsToSet, depth);
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename BoxIteratorPair>
inline NrBoxes ImplicitBoxTree<A, DIM, REAL, N>::changeFlagsByIterators(BoxIteratorPair points,
                                                                        Flags flagsToUnset, Flags flagsToSet,
                                                                        Depth depth)
{
  return this->changeFlagsBySearch(this->searchCompacted(points, depth), flagsToUnset, flagsToSet, depth);
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline NrBoxes ImplicitBoxTree<A, DIM, REAL, N>::changeFlagsByPoints(
    const typename ThrustSystem<A>::Vector<REAL>& points, Flags flagsToUnset, Flags flagsToSet, Depth depth)
{
  auto ti_begin =
      thrust::make_transform_iterator(
          thrust::make_transform_iterator(
              thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                              thrust::placeholders::_1 * NrPoints(DIM)),
              AdvanceConstRealPointerFunctor<REAL>(thrust::raw_pointer_cast(points.data()))),
          ComputeBoxFromPointFunctor<DIM, REAL, N>(thrust::raw_pointer_cast(center_.data()), 
                                                   thrust::raw_pointer_cast(radius_.data()), 
                                                   thrust::raw_pointer_cast(sdScheme_.data()),
                                                   thrust::raw_pointer_cast(sdCount_.data()),
                                                   this->isSdSchemeDefault(),
                                                   Flag::NONE));
  
  return this->changeFlagsByIterators(thrust::make_pair(ti_begin, ti_begin + points.size() / DIM),
                                      flagsToUnset, flagsToSet, depth);
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline NrBoxes ImplicitBoxTree<A, DIM, REAL, N>::changeFlagsBySearch(
    const typename ThrustSystem<A>::Vector<NrBoxes>& searchVector, Flags flagsToUnset, Flags flagsToSet, Depth depth)
{
  // if last element is getInvalidNrBoxes(), then reduce later the end()-iterator
  NrBoxes isLastElementInvalid = searchVector.size() ? searchVector.back() == getInvalidNrBoxes() : 0;
    
  if (depth == 0) {
    
    bool isSurroundingBoxHit = thrust::any_of(typename ThrustSystem<A>::execution_policy(),
                                              searchVector.begin(), searchVector.end() - isLastElementInvalid,
                                              thrust::placeholders::_1 == NrBoxes(0));
    
    bool haveAllBoxesFlagsSet = thrust::all_of(typename ThrustSystem<A>::execution_policy(),
                                               flagsVector_.begin(), flagsVector_.end(),
                                               AreFlagsSetFunctor(flagsToUnset));
    
    if (isSurroundingBoxHit && haveAllBoxesFlagsSet) {
      
      thrust::transform(typename ThrustSystem<A>::execution_policy(),
                        flagsVector_.begin(), flagsVector_.end(),
                        flagsVector_.begin(),
                        ChangeFlagsFunctor(flagsToUnset, flagsToSet));
      
      if (isAnyFlagSet(flagsToCheckForLeafIndices_, flagsToUnset) ||
          isAnyFlagSet(flagsToCheckForLeafIndices_, flagsToSet)) {
        this->freeLeafIndicesFromDepth();
      }
      
      return 1;
      
    } else {
      
      return 0;
      
    }
    
  } else if (depth == getLeafDepth() || this->areAllLeavesOnDepth(depth)) {
    
    // count boxes that have Flags flagsToUnset not set
    NrBoxes res = thrust::count_if(typename ThrustSystem<A>::execution_policy(),
                                   thrust::make_permutation_iterator(flagsVector_.begin(),
                                                                     searchVector.begin()), // begin
                                   thrust::make_permutation_iterator(flagsVector_.begin(),
                                                                     searchVector.end() - isLastElementInvalid), // end
                                   AreFlagsSetFunctor(flagsToUnset));
    
    thrust::transform_if(typename ThrustSystem<A>::execution_policy(),
                         thrust::make_permutation_iterator(flagsVector_.begin(), searchVector.begin()), // begin
                         thrust::make_permutation_iterator(flagsVector_.begin(),
                                                           searchVector.end() - isLastElementInvalid), // end
                         searchVector.begin(), // stencil
                         thrust::make_permutation_iterator(flagsVector_.begin(),
                                                           searchVector.begin()), // in-place trafo
                         ChangeFlagsFunctor(flagsToUnset, flagsToSet), // operation to execute
                         thrust::placeholders::_1 != getInvalidNrBoxes()); // predicate for stencil
    
    if (res > 0 && (isAnyFlagSet(flagsToCheckForLeafIndices_, flagsToUnset) || 
                    isAnyFlagSet(flagsToCheckForLeafIndices_, flagsToSet))) {
      this->freeLeafIndicesFromDepth();
    }
    
    return res;
    
  } else if (depth == getUnsubdividableDepth() || depth <= this->depth()) {
    
    // fill stencilVector with "false"
    typename ThrustSystem<A>::Vector<bool> stencilVector(this->count(depth), false);
    
    thrust::transform_if(typename ThrustSystem<A>::execution_policy(),
                         thrust::make_permutation_iterator(stencilVector.begin(), searchVector.begin()), // begin
                         thrust::make_permutation_iterator(stencilVector.begin(),
                                                           searchVector.end() - isLastElementInvalid), // end
                         searchVector.begin(), // stencil for trafo
                         thrust::make_permutation_iterator(stencilVector.begin(),
                                                           searchVector.begin()), // in-place trafo
                         thrust::logical_not<bool>(), // stencilVector (false) -> not(stencilVector) (true)
                         thrust::placeholders::_1 != getInvalidNrBoxes()); // predicate for stencil
    
    NrBoxes res = this->changeFlagsByStencil(stencilVector.begin(), flagsToUnset, flagsToSet, depth);
    
    return res;
    
  } else {
    
    return 0;
    
  }
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename BoolIterator>
inline NrBoxes ImplicitBoxTree<A, DIM, REAL, N>::changeFlagsByStencil(BoolIterator stencil_begin,
                                                                      Flags flagsToUnset, Flags flagsToSet,
                                                                      Depth depth)
{
  NrBoxes nBoxes = this->countIfFlags(AreFlagsSetFunctor(flagsToUnset), FlagsAndFunctor(), stencil_begin, depth);
  
  if (depth == 0) {
    
    if (*stencil_begin) { // must stand inside then-branch!
      thrust::transform(typename ThrustSystem<A>::execution_policy(),
                        flagsVector_.begin(), flagsVector_.end(),
                        flagsVector_.begin(),
                        ChangeFlagsFunctor(flagsToUnset, flagsToSet));
    }
    
  } else if (depth == getLeafDepth() || this->areAllLeavesOnDepth(depth)) {
    
    thrust::transform_if(typename ThrustSystem<A>::execution_policy(),
                         flagsVector_.begin(), flagsVector_.end(),
                         stencil_begin, // stencil
                         flagsVector_.begin(), // in-place trafo
                         ChangeFlagsFunctor(flagsToUnset, flagsToSet),
                         thrust::identity<bool>()); // predicate for stencil
    
  } else if (depth == getUnsubdividableDepth()) {
    
    // initialise boxes' indeces in decremented Depth if needed
    this->initializeIndicesInDepth(getUnsubdividableDepth());
    
    // pi_begin[i] == stencil_begin[indicesInDepth_[i]]
    auto pi_begin = thrust::make_permutation_iterator(stencil_begin, indicesInDepth_.begin());
    
    // bool-vector indicating whether all boxes that belong to the same box in Depth getUnsubdividableDepth()
    // have the Flags flagsToUnset set
    typename ThrustSystem<A>::Vector<bool> haveAllBoxesInSegmentFlagsSet(this->size());
    
    thrust::transform(typename ThrustSystem<A>::execution_policy(),
                      flagsVector_.begin(), flagsVector_.end(),
                      haveAllBoxesInSegmentFlagsSet.begin(),
                      AreFlagsSetFunctor(flagsToUnset));
    
    // iterator with index in Depth getUnsubdividableDepth() and indicator whether the box is unsubdividable
    auto indicesInDepth_bool_begin = thrust::make_zip_iterator(thrust::make_tuple(indicesInDepth_.begin(),
                                                                                  this->isUnsubdividable_begin()));
    
    // last haveAllBoxesInSegmentFlagsSet-element in segment indicates 
    // whether the corresponding box in Depth getUnsubdividableDepth() has the Flags flagsToUnset set
    thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(),
                                  indicesInDepth_bool_begin, indicesInDepth_bool_begin + this->size(), // keys
                                  haveAllBoxesInSegmentFlagsSet.begin(), // values to "sum up"
                                  haveAllBoxesInSegmentFlagsSet.begin(), // output
                                  thrust::equal_to<thrust::tuple<NrBoxes, bool>>(), // compare-operator for keys
                                  thrust::logical_and<bool>()); // "sum" operation
                    
    // only "going" back and forth in the same segment guarantees the "right" boolean for all its boxes
    // i.e. both inclusive_scan_by_key operations together act like a "segmented_all_of"
    thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(),
                                  thrust::make_reverse_iterator(indicesInDepth_bool_begin + this->size()),// keys_begin
                                  thrust::make_reverse_iterator(indicesInDepth_bool_begin), // keys_end
                                  haveAllBoxesInSegmentFlagsSet.rbegin(), // Note: all reverse_iterators
                                  haveAllBoxesInSegmentFlagsSet.rbegin(), // output
                                  thrust::equal_to<thrust::tuple<NrBoxes, bool>>(), // compare-operator for keys
                                  thrust::logical_and<bool>()); // "sum" operation
    
    // store values since stencil and in-/output range would otherwise overlap in later transform_if
    typename ThrustSystem<A>::Vector<bool> isUnsubdividableVector(this->isUnsubdividable_begin(),
                                                                  this->isUnsubdividable_begin() + this->size());
    
    // fusing stencil, isUnsubdividable_begin and haveAllBoxesInSegmentFlagsSet
    auto zi_begin = thrust::make_zip_iterator(thrust::make_tuple(pi_begin,
                                                                 isUnsubdividableVector.begin(),
                                                                 haveAllBoxesInSegmentFlagsSet.begin()));
    
    // change Flags in boxes b if b is unsubdividable and stencil_begin[indicesInDepth_[*b]] is true
    thrust::transform_if(typename ThrustSystem<A>::execution_policy(),
                         flagsVector_.begin(), flagsVector_.end(),
                         zi_begin, // stencil
                         flagsVector_.begin(), // in-place trafo
                         ChangeFlagsFunctor(flagsToUnset, flagsToSet),
                         AllIn3TupleFunctor()); // logical and in tuple of stencil; predicate for stencil
    
  } else if (depth <= this->depth()) {
    
    // initialise boxes' indeces in decremented Depth if needed
    this->initializeIndicesInDepth(depth);
    
    // pi_begin[i] == stencil_begin[indicesInDepth_[i]]
    auto pi_begin = thrust::make_permutation_iterator(stencil_begin, indicesInDepth_.begin());
    
    // depth_begin[i] == hasAtLeastDepth(this[i])
    auto depth_begin = thrust::make_transform_iterator(this->begin(), HasAtLeastDepthFunctor<N>(depth));
    
    // bool-vector indicating whether all boxes that belong to the same box in Depth depth 
    // have the Flags flagsToUnset set
    typename ThrustSystem<A>::Vector<bool> haveAllBoxesInSegmentFlagsSet(this->size());
    
    thrust::transform(typename ThrustSystem<A>::execution_policy(),
                      flagsVector_.begin(), flagsVector_.end(),
                      haveAllBoxesInSegmentFlagsSet.begin(),
                      AreFlagsSetFunctor(flagsToUnset));
    
    // iterator with index in Depth depth and indicator whether the box has at least Depth depth
    auto indicesInDepth_bool_begin = thrust::make_zip_iterator(thrust::make_tuple(indicesInDepth_.begin(),
                                                                                  depth_begin));
    
    // last haveAllBoxesInSegmentFlagsSet-element in segment indicates 
    // whether the corresponding box in Depth depth has the Flags flagsToUnset set
    thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(),
                                  indicesInDepth_bool_begin, indicesInDepth_bool_begin + this->size(), // keys
                                  haveAllBoxesInSegmentFlagsSet.begin(), // values to "sum up"
                                  haveAllBoxesInSegmentFlagsSet.begin(), // output
                                  thrust::equal_to<thrust::tuple<NrBoxes, bool>>(), // compare-operator for keys
                                  thrust::logical_and<bool>()); // "sum" operation
                    
    // only "going" back and forth in the same segment guarantees the "right" boolean for all its boxes
    // i.e. both inclusive_scan_by_key operations together act like a "segmented_all_of"
    thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(),
                                  thrust::make_reverse_iterator(indicesInDepth_bool_begin + this->size()),// keys_begin
                                  thrust::make_reverse_iterator(indicesInDepth_bool_begin), // keys_end
                                  haveAllBoxesInSegmentFlagsSet.rbegin(), // Note: all reverse_iterators
                                  haveAllBoxesInSegmentFlagsSet.rbegin(), // output
                                  thrust::equal_to<thrust::tuple<NrBoxes, bool>>(), // compare-operator for keys
                                  thrust::logical_and<bool>()); // "sum" operation
    
    // fusing stencil, hasAtLeastDepth and haveAllBoxesInSegmentFlagsSet
    auto zi_begin = thrust::make_zip_iterator(thrust::make_tuple(pi_begin,
                                                                 depth_begin,
                                                                 haveAllBoxesInSegmentFlagsSet.begin()));
    
    // change Flags in boxes b if b has at least Depth depth and stencil_begin[indicesInDepth_[*b]] is true
    thrust::transform_if(typename ThrustSystem<A>::execution_policy(),
                         flagsVector_.begin(), flagsVector_.end(),
                         zi_begin, // stencil
                         flagsVector_.begin(), // in-place trafo
                         ChangeFlagsFunctor(flagsToUnset, flagsToSet),
                         AllIn3TupleFunctor()); // logical and in tuple of stencil; predicate for stencil
  }
  
  if (nBoxes > 0 && (isAnyFlagSet(flagsToCheckForLeafIndices_, flagsToUnset) || 
                     isAnyFlagSet(flagsToCheckForLeafIndices_, flagsToSet))) {
    this->freeLeafIndicesFromDepth();
  }
  
  return nBoxes;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline NrBoxes ImplicitBoxTree<A, DIM, REAL, N>::count(Depth depth) const
{
  if (depth == 0) {
    
    return 1;
    
  } else if (depth == getLeafDepth() || this->areAllLeavesOnDepth(depth)) {
    
    return this->size();
    
  } else if (depth == getUnsubdividableDepth()) {
    
    // iterators indicating whether a box is unsubdividable and it is the first sibling (if there are two)
    auto zi = thrust::make_zip_iterator(thrust::make_tuple(this->isNewBoxInDepth_begin(getUnsubdividableDepth()),
                                                           this->isUnsubdividable_begin()));
    
    return thrust::count_if(typename ThrustSystem<A>::execution_policy(),
                            zi, zi + this->size(),
                            AllIn2TupleFunctor());
    
  } else if (depth <= this->depth()) {
    
    // iterator indicating where a new box in decremented Depth starts
    auto ti = this->isNewBoxInDepth_begin(depth);
    
    return thrust::count_if(typename ThrustSystem<A>::execution_policy(),
                            ti, ti + this->size(),
                            thrust::identity<bool>());
    
  } else {
    
    return 0;
    
  }
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename Predicate, typename Combination>
inline NrBoxes ImplicitBoxTree<A, DIM, REAL, N>::countIfFlags(Predicate pred, Combination comb, Depth depth)
{
  return this->countIfFlags(pred, comb, thrust::make_constant_iterator(true), depth);
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename Predicate, typename Combination, typename BoolIterator>
inline NrBoxes ImplicitBoxTree<A, DIM, REAL, N>::countIfFlags(Predicate pred, Combination comb,
                                                              BoolIterator stencil_begin, Depth depth)
{
  if (depth == 0) {
    
    Flags flagsOfSurroundingBox = thrust::reduce(typename ThrustSystem<A>::execution_policy(),
                                                 flagsVector_.begin(), flagsVector_.end(), // values to reduce
                                                 Flags(flagsVector_.front()), // init value
                                                 comb); // "sum"-operation
    
    return *stencil_begin && pred(flagsOfSurroundingBox);
    
  } else if (depth == getLeafDepth() || this->areAllLeavesOnDepth(depth)) {
    
    // iterator indicating whether a box fulfills pred and its corresponding stencil entry
    auto zi_begin =
        thrust::make_zip_iterator(
            thrust::make_tuple(
                thrust::make_transform_iterator(flagsVector_.begin(), pred),
                stencil_begin));
    
    return thrust::count_if(typename ThrustSystem<A>::execution_policy(),
                            zi_begin, zi_begin + this->count(getLeafDepth()),
                            AllIn2TupleFunctor()); // logical and in tuple of zi_begin
    
  } else if (depth == getUnsubdividableDepth()) {
    
    // initialise boxes' indeces in decremented Depth
    // iterator indicating where a new box in decremented Depth getUnsubdividableDepth() "starts"
    auto ti = this->initializeIndicesInDepth(getUnsubdividableDepth());
    
    // iterator with index in decremented Depth and indicator if box in unsubdividable
    auto indicesInDepth_bool_begin = thrust::make_zip_iterator(thrust::make_tuple(indicesInDepth_.begin(),
                                                                                  this->isUnsubdividable_begin()));
    
    // _flagsVector[ti==1] gives Flags of (the box in decr. depth, i.e. all its leaves), 
    // but also Flags of (box with Depth < depth)
    typename ThrustSystem<A>::Vector<Flags> _flagsVector(this->size());
    
    thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(), // Note: all reverse_iterators
                                  thrust::make_reverse_iterator(indicesInDepth_bool_begin + this->size()),// keys_begin
                                  thrust::make_reverse_iterator(indicesInDepth_bool_begin), // keys_end
                                  flagsVector_.rbegin(), // values to "sum up"
                                  _flagsVector.rbegin(), // output
                                  thrust::equal_to<thrust::tuple<NrBoxes, bool>>(), // operator==(…)
                                  comb); // "sum"-operation
    
    // fuse _flagsVector with pred
    auto flags_begin = thrust::make_transform_iterator(_flagsVector.begin(), pred);
    
    // pi_begin[i] == stencil_begin[indicesInDepth_[i]]
    auto pi_begin = thrust::make_permutation_iterator(stencil_begin, indicesInDepth_.begin());
    
    // fuse all 4 iterators to indicate where a new box in Depth depth starts, is unsubdividable,
    // and whose all leaves fulfill pred and indicating stencil
    auto zi_begin = thrust::make_zip_iterator(thrust::make_tuple(ti,
                                                                 this->isUnsubdividable_begin(),
                                                                 flags_begin,
                                                                 pi_begin));
    
    return thrust::count_if(typename ThrustSystem<A>::execution_policy(),
                            zi_begin, zi_begin + this->size(),
                            AllIn4TupleFunctor()); // logical and in tuple of zi_begin
    
  } else if (depth <= this->depth()) {
    
    // initialise boxes' indeces in decremented Depth
    // iterator indicating where a new box in decremented Depth depth (omitting smaller depth) "starts"
    auto ti = this->initializeIndicesInDepth(depth);
    
    // depth_begin[i] == hasAtLeastDepth(this[i])
    auto depth_begin = thrust::make_transform_iterator(this->begin(), HasAtLeastDepthFunctor<N>(depth));
    
    // iterator with index in decremented Depth and indicator of appropriate depth
    auto indicesInDepth_bool_begin = thrust::make_zip_iterator(thrust::make_tuple(indicesInDepth_.begin(),
                                                                                  depth_begin));
    
    // _flagsVector[ti==1] gives Flags of (the box in decr. depth, i.e. all its leaves), 
    // but also Flags of (box with Depth < depth)
    typename ThrustSystem<A>::Vector<Flags> _flagsVector(this->size());
    // Note all reverse_iterators!
    thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(),
                                  thrust::make_reverse_iterator(indicesInDepth_bool_begin + this->size()),// keys_begin
                                  thrust::make_reverse_iterator(indicesInDepth_bool_begin), // keys_end
                                  flagsVector_.rbegin(), // values to "sum up"
                                  _flagsVector.rbegin(), // output
                                  thrust::equal_to<thrust::tuple<NrBoxes, bool>>(), // operator==(…)
                                  comb); // "sum"-operation
    
    // fuse _flagsVector with pred
    auto flags_begin = thrust::make_transform_iterator(_flagsVector.begin(), pred);
    
    // pi_begin[i] == stencil_begin[indicesInDepth_[i]]
    auto pi_begin = thrust::make_permutation_iterator(stencil_begin, indicesInDepth_.begin());
    
    // fuse all 3 iterators to indicate where a new box in Depth depth starts (i.e. has at least Depth depth)
    // and whose all leaves fulfill pred and indicating stencil
    auto zi_begin = thrust::make_zip_iterator(thrust::make_tuple(ti, flags_begin, pi_begin));
    
    return thrust::count_if(typename ThrustSystem<A>::execution_policy(),
                            zi_begin, zi_begin + this->size(),
                            AllIn3TupleFunctor()); // logical and in tuple of zi_begin
    
  } else {
    
    return 0;
    
  }
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline NrBoxes ImplicitBoxTree<A, DIM, REAL, N>::deleteDepth(Depth depth)
{
  if (depth == 0) {
    
    this->resize(0);
    this->insertSurroundingBoxIfEmpty();
    
    this->freeIndicesInDepth();
    this->freeLeafIndicesFromDepth();
    
    return 1;
    
  } else if (depth == getLeafDepth() || depth == getUnsubdividableDepth() || depth > this->depth()) {
    
    return 0;
    
  } else {
    
    NrBoxes nBoxes = this->count(depth);
    
    thrust::transform(typename ThrustSystem<A>::execution_policy(),
                      this->begin(), this->end(),
                      this->begin(), // in-place trafo
                      DecrementDepthToFunctor<N>(depth - Depth(1))); // reduce depth by 1
    
    // first box of segment gets intersection of all flags in the segment
    thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(),
                                  this->rbegin(), this->rend(), // keys
                                  flagsVector_.rbegin(), // Note all reverse_iterators!
                                  flagsVector_.rbegin(), // in-place trafo
                                  IsSameBoxFunctor<N>(), // "segment operator" for keys
                                  FlagsAndFunctor());
    
    // removes all but the first element of a group of equal boxes
    auto box_new_end = thrust::unique(typename ThrustSystem<A>::execution_policy(),
                                      this->begin(), this->end(),
                                      IsSameBoxFunctor<N>());
    
    this->resize(box_new_end - this->begin());
    
    this->freeIndicesInDepth();
    this->freeLeafIndicesFromDepth();
    
    return nBoxes;
  }
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline Depth ImplicitBoxTree<A, DIM, REAL, N>::depth() const
{
  return thrust::reduce(typename ThrustSystem<A>::execution_policy(),
                        depthVector_.begin(), depthVector_.end(),
                        Depth(0),
                        thrust::maximum<Depth>());
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline typename ThrustSystem<A>::Vector<NrBoxes> ImplicitBoxTree<A, DIM, REAL, N>::enumerate() const
{
  typename ThrustSystem<A>::Vector<NrBoxes> res(this->depth() + 1);
  
  for (Depth i = 0; i <= this->depth(); ++i) {
    res[i] = this->count(i);
  }
  
  return res;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline void ImplicitBoxTree<A, DIM, REAL, N>::freeIndicesInDepth()
{
  indicesInDepth_.clear(); // resizes vector to 0
  indicesInDepth_.shrink_to_fit(); // reduces capacity to 0
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline void ImplicitBoxTree<A, DIM, REAL, N>::freeLeafIndicesFromDepth()
{
  leafIndicesFromDepth_.clear(); // resizes vector to 0
  leafIndicesFromDepth_.shrink_to_fit(); // reduces capacity to 0
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<Architecture A2, Dimension DIM2, typename REAL2, Depth N2>
inline bool ImplicitBoxTree<A, DIM, REAL, N>::hasSameBoxes(
    const ImplicitBoxTree<A2, DIM2, REAL2, N2>& implicitBoxTree) const
{
  bool res;
  
  if (A == A2) {
    typename ThrustSystem<A>::Vector<BitPattern<N>> _bitPatternVector(implicitBoxTree.bitPatternVector_.begin(),
                                                                      implicitBoxTree.bitPatternVector_.end());
    res = DIM == DIM2 &&
          thrust::equal(typename ThrustSystem<A>::execution_policy(),
                        center_.begin(), center_.end(),
                        implicitBoxTree.center_.begin()) &&
          thrust::equal(typename ThrustSystem<A>::execution_policy(),
                        radius_.begin(), radius_.end(),
                        implicitBoxTree.radius_.begin()) &&
          thrust::equal(typename ThrustSystem<A>::execution_policy(),
                        bitPatternVector_.begin(), bitPatternVector_.end(),
                        _bitPatternVector.begin()) &&
          thrust::equal(typename ThrustSystem<A>::execution_policy(),
                        depthVector_.begin(), depthVector_.end(),
                        implicitBoxTree.depthVector_.begin()) &&
          thrust::equal(typename ThrustSystem<A>::execution_policy(),
                        flagsVector_.begin(), flagsVector_.end(),
                        implicitBoxTree.flagsVector_.begin()) &&
          thrust::equal(typename ThrustSystem<A>::execution_policy(),
                        sdScheme_.begin(), sdScheme_.begin() + this->depth(),
                        implicitBoxTree.sdScheme_.begin());
  } else {
    res = DIM == DIM2;
    
    if (res) {
      typename ThrustSystem<A>::Vector<REAL2> _center(implicitBoxTree.center_.begin(),
                                                      implicitBoxTree.center_.end());
      res = thrust::equal(typename ThrustSystem<A>::execution_policy(),
                          center_.begin(), center_.end(),
                          _center.begin());
    }
    
    if (res) {
      typename ThrustSystem<A>::Vector<REAL2> _radius(implicitBoxTree.radius_.begin(),
                                                      implicitBoxTree.radius_.end());
      res = thrust::equal(typename ThrustSystem<A>::execution_policy(),
                          radius_.begin(), radius_.end(),
                          _radius.begin());
    }
    
    if (res) {
      typename ThrustSystem<A>::Vector<BitPattern<N>> _bitPatternVector(implicitBoxTree.bitPatternVector_.begin(),
                                                                        implicitBoxTree.bitPatternVector_.end());
      res = thrust::equal(typename ThrustSystem<A>::execution_policy(),
                          bitPatternVector_.begin(), bitPatternVector_.end(),
                          _bitPatternVector.begin());
    }
    
    if (res) {
      typename ThrustSystem<A>::Vector<Depth> _depthVector(implicitBoxTree.depthVector_.begin(),
                                                           implicitBoxTree.depthVector_.end());
      res = thrust::equal(typename ThrustSystem<A>::execution_policy(),
                          depthVector_.begin(), depthVector_.end(),
                          _depthVector.begin());
    }
    
    if (res) {
      typename ThrustSystem<A>::Vector<Flags> _flagsVector(implicitBoxTree.flagsVector_.begin(),
                                                           implicitBoxTree.flagsVector_.end());
      res = thrust::equal(typename ThrustSystem<A>::execution_policy(),
                          flagsVector_.begin(), flagsVector_.end(),
                          _flagsVector.begin());
    }
    
    if (res) {
      typename ThrustSystem<A>::Vector<Dimension> _sdScheme(implicitBoxTree.sdScheme_.begin(),
                                                            implicitBoxTree.sdScheme_.end());
      res = thrust::equal(typename ThrustSystem<A>::execution_policy(),
                          sdScheme_.begin(), sdScheme_.begin() + this->depth(),
                          _sdScheme.begin());
    }
  }
  
  return res;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline thrust::transform_iterator<
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
ImplicitBoxTree<A, DIM, REAL, N>::initializeIndicesInDepth(Depth depth)
{
  // iterator indicating where a new box in decremented Depth depth "starts"
  auto ti = this->isNewBoxInDepth_begin(depth);
  
  if (depth != depthOfIndices_ || indicesInDepth_.size() == 0) {
    
    indicesInDepth_.resize(this->size());
    
    if (depth == 0) {
      
      thrust::fill(typename ThrustSystem<A>::execution_policy(),
                   indicesInDepth_.begin(), indicesInDepth_.end(),
                   NrBoxes(0));
      
    } else if (depth == getLeafDepth() || this->areAllLeavesOnDepth(depth)) {
      
      thrust::sequence(typename ThrustSystem<A>::execution_policy(),
                       indicesInDepth_.begin(), indicesInDepth_.end());
      
    } else if (depth == getUnsubdividableDepth()) {
      
      // iterators indicating whether a box is unsubdividable and it is the first sibling (if there are two)
      auto zi = thrust::make_zip_iterator(thrust::make_tuple(ti,
                                                             this->isUnsubdividable_begin()));
      
      // first copy for conversion bool->NrBoxes; otherwise wrong results after inclusive_scan on GPU
      thrust::transform(typename ThrustSystem<A>::execution_policy(),
                        zi, zi + this->size(),
                        indicesInDepth_.begin(),
                        AllIn2TupleFunctor());
      
      thrust::inclusive_scan(typename ThrustSystem<A>::execution_policy(),
                             indicesInDepth_.begin(), indicesInDepth_.end(),
                             indicesInDepth_.begin());
      
      // first box with Depth >= depth has index 1 until now
      // => reduce all indices by 1, except for eventually existing first boxes with Depth < depth
      // (so that their index stays 0)
      thrust::transform_if(typename ThrustSystem<A>::execution_policy(),
                           indicesInDepth_.begin(), indicesInDepth_.end(),
                           indicesInDepth_.begin(), // in-place trafo
                           thrust::placeholders::_1 - NrBoxes(1), // decrement index
                           thrust::placeholders::_1 > NrBoxes(0));
      
    } else if (depth <= this->depth()) {
      
      // first copy for conversion bool->NrBoxes; otherwise wrong results after inclusive_scan on GPU
      thrust::copy(typename ThrustSystem<A>::execution_policy(),
                   ti, ti + this->size(),
                   indicesInDepth_.begin());
      
      thrust::inclusive_scan(typename ThrustSystem<A>::execution_policy(),
                             indicesInDepth_.begin(), indicesInDepth_.end(),
                             indicesInDepth_.begin());
      
      // first box with Depth >= depth has index 1 until now
      // => reduce all indices by 1, except for eventually existing first boxes with Depth < depth
      // (so that their index stays 0)
      thrust::transform_if(typename ThrustSystem<A>::execution_policy(),
                           indicesInDepth_.begin(), indicesInDepth_.end(),
                           indicesInDepth_.begin(), // in-place trafo
                           thrust::placeholders::_1 - NrBoxes(1), // decrement index
                           thrust::placeholders::_1 > NrBoxes(0));
    } else {
      
      this->freeIndicesInDepth();
      
    }
    
    depthOfIndices_ = depth;
  }
  
  return ti;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline void ImplicitBoxTree<A, DIM, REAL, N>::initializeLeafIndicesFromDepth(Depth depth, Flags flagsToCheck)
{
  if (depth != depthOfLeafIndices_ ||
      flagsToCheck != flagsToCheckForLeafIndices_ || 
      leafIndicesFromDepth_.size() == 0) {
    
    if (depth == 0) {
      
      leafIndicesFromDepth_.resize(1);
      leafIndicesFromDepth_.front() = 0;
      
      // if not all boxes have flagsToCheck set, the root has it neither (areFlagsSet(Flag::NONE) is always true)
      if (flagsToCheck != Flag::NONE && !thrust::all_of(typename ThrustSystem<A>::execution_policy(),
                                                        flagsVector_.begin(), flagsVector_.end(),
                                                        AreFlagsSetFunctor(flagsToCheck))) {
        leafIndicesFromDepth_.resize(0);
      }
      
    } else if (depth == getLeafDepth() || this->areAllLeavesOnDepth(depth)) {
      
      leafIndicesFromDepth_.resize(this->size());
      
      if (flagsToCheck == Flag::NONE) {
        // reduce computing time since areFlagsSet(Flag::NONE) is always true
        thrust::sequence(typename ThrustSystem<A>::execution_policy(),
                         leafIndicesFromDepth_.begin(), leafIndicesFromDepth_.end());
      } else {
        auto new_end = thrust::copy_if(typename ThrustSystem<A>::execution_policy(),
                                       thrust::make_counting_iterator(NrBoxes(0)), // values_begin
                                       thrust::make_counting_iterator(NrBoxes(this->size())), // values_end
                                       flagsVector_.begin(), // stencil
                                       leafIndicesFromDepth_.begin(), // resulting range
                                       AreFlagsSetFunctor(flagsToCheck)); // predicate for stencil
        
        leafIndicesFromDepth_.resize(new_end - leafIndicesFromDepth_.begin());
        leafIndicesFromDepth_.shrink_to_fit();
      }
      
    } else if (depth == getUnsubdividableDepth()) {
      
      leafIndicesFromDepth_.resize(this->size());
      
      if (flagsToCheck == Flag::NONE) {
        auto new_end = thrust::copy_if(typename ThrustSystem<A>::execution_policy(),
                                       thrust::make_counting_iterator(NrBoxes(0)), // values_begin
                                       thrust::make_counting_iterator(NrBoxes(this->size())), // values_end
                                       thrust::make_zip_iterator(
                                           thrust::make_tuple(
                                               this->isNewBoxInDepth_begin(getUnsubdividableDepth()),
                                               this->isUnsubdividable_begin())), // stencil for copying
                                       leafIndicesFromDepth_.begin(), // resulting range
                                       AllIn2TupleFunctor()); // predicate for stencil
        
        leafIndicesFromDepth_.resize(new_end - leafIndicesFromDepth_.begin());
        leafIndicesFromDepth_.shrink_to_fit();
      } else {
        typename ThrustSystem<A>::Vector<bool> areFlagsSetVector(this->size());
        // areFlagsSetVector gives the booleans whether a box in Depth depth has Flags flagsToCheck set
        thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(), // Note: all reverse_iterators!
                                      thrust::make_transform_iterator(
                                          this->rbegin(), DecrementDepthByFunctor<N>(1)), // keys_rbegin
                                      thrust::make_transform_iterator(
                                          this->rend(), DecrementDepthByFunctor<N>(1)), // keys_rend
                                      thrust::make_transform_iterator(
                                          flagsVector_.rbegin(), AreFlagsSetFunctor(flagsToCheck)), // values_rbegin
                                      areFlagsSetVector.rbegin(), // output
                                      IsSameBoxFunctor<N>(), // "segment operator" for keys
                                      thrust::logical_and<bool>()); // "sum" operation for values
        
        auto new_end = thrust::copy_if(typename ThrustSystem<A>::execution_policy(),
                                       thrust::make_counting_iterator(NrBoxes(0)), // values_begin
                                       thrust::make_counting_iterator(NrBoxes(this->size())), // values_end
                                       thrust::make_zip_iterator(
                                           thrust::make_tuple( // stencil for copying
                                               this->isNewBoxInDepth_begin(getUnsubdividableDepth()),
                                               this->isUnsubdividable_begin(), // new box in Depth depth
                                               areFlagsSetVector.begin())), // has new box flagsToCheck set
                                       leafIndicesFromDepth_.begin(), // resulting range
                                       AllIn3TupleFunctor()); // predicate for stencil
        
        leafIndicesFromDepth_.resize(new_end - leafIndicesFromDepth_.begin());
        leafIndicesFromDepth_.shrink_to_fit();
      }
      
    } else if (depth <= this->depth()) {
      
      leafIndicesFromDepth_.resize(this->size());
      
      if (flagsToCheck == Flag::NONE) {
        auto new_end = thrust::copy_if(typename ThrustSystem<A>::execution_policy(),
                                       thrust::make_counting_iterator(NrBoxes(0)), // values_begin
                                       thrust::make_counting_iterator(NrBoxes(this->size())), // values_end
                                       this->isNewBoxInDepth_begin(depth), // stencil for copying
                                       leafIndicesFromDepth_.begin(), // resulting range
                                       thrust::identity<bool>()); // predicate for stencil
        
        leafIndicesFromDepth_.resize(new_end - leafIndicesFromDepth_.begin());
        leafIndicesFromDepth_.shrink_to_fit();
      } else {
        typename ThrustSystem<A>::Vector<bool> areFlagsSetVector(this->size());
        // areFlagsSetVector[this->isNewBoxInDepth_begin(depth) == true] gives the boolean whether the box in 
        // Depth depth has Flags flagsToCheck set
        thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(), // Note: all reverse_iterators!
                                      thrust::make_transform_iterator(
                                          this->rbegin(), DecrementDepthToFunctor<N>(depth)), // keys_rbegin
                                      thrust::make_transform_iterator(
                                          this->rend(), DecrementDepthToFunctor<N>(depth)), // keys_rend
                                      thrust::make_transform_iterator(
                                          flagsVector_.rbegin(), AreFlagsSetFunctor(flagsToCheck)), // values_rbegin
                                      areFlagsSetVector.rbegin(), // output
                                      IsSameBoxFunctor<N>(), // "segment operator" for keys
                                      thrust::logical_and<bool>()); // "sum" operation for values
        
        auto new_end = thrust::copy_if(typename ThrustSystem<A>::execution_policy(),
                                       thrust::make_counting_iterator(NrBoxes(0)), // values_begin
                                       thrust::make_counting_iterator(NrBoxes(this->size())), // values_end
                                       thrust::make_zip_iterator(thrust::make_tuple( // stencil for copying
                                           this->isNewBoxInDepth_begin(depth), // new box in Depth depth
                                           areFlagsSetVector.begin())), // has new box flagsToCheck set
                                       leafIndicesFromDepth_.begin(), // resulting range
                                       AllIn2TupleFunctor()); // predicate for stencil
        
        leafIndicesFromDepth_.resize(new_end - leafIndicesFromDepth_.begin());
        leafIndicesFromDepth_.shrink_to_fit();
      }
      
    } else {
      
      this->freeLeafIndicesFromDepth();
      
    }
    
    depthOfLeafIndices_ = depth;
    flagsToCheckForLeafIndices_ = flagsToCheck;
  }
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename BoxIteratorPair, typename BoolIterator>
inline void ImplicitBoxTree<A, DIM, REAL, N>::insertByIterators(BoxIteratorPair points,
                                                                Depth depth, Flags flagsForInserted, Flags flagsForHit,
                                                                BoolIterator res_begin)
{
  NrPoints nPoints = points.second - points.first;
  
  if (depth == 0) {
    
    // nothing to insert since surroundig box always exists
    thrust::fill_n(typename ThrustSystem<A>::execution_policy(), res_begin, nPoints, false);
    
    bool areAllPointsInvalid = thrust::all_of(typename ThrustSystem<A>::execution_policy(),
                                              points.first, points.second,
                                              IsInvalidBoxFunctor<N>());
    
    // set flags if at least one point is in the surroundig box
    if (flagsForHit != Flag::NONE && !areAllPointsInvalid) {
      thrust::transform(typename ThrustSystem<A>::execution_policy(),
                        flagsVector_.begin(), flagsVector_.end(),
                        flagsVector_.begin(),
                        SetFlagsFunctor(flagsForHit));
    }
    
  } else if (depth == getLeafDepth() || depth <= getMaxDepth<N>()) {
    
    auto res_temp_begin = res_begin;
    auto point_temp_begin = points.first;
    
    // allocate other box storage so that the ImplicitBoxTree need not to be increased so that 
    // this stays sorted and therefore this->search(inserting boxes) still works
    typename ThrustSystem<A>::Vector<BitPattern<N>> _bitPatternVector;
    typename ThrustSystem<A>::Vector<Depth> _depthVector;
    typename ThrustSystem<A>::Vector<Flags> _flagsVector;
    
    while (points.second - point_temp_begin > 0) {
      
      // a quarter to "ensure" enough space for later search, sort, and unique
      NrBoxes nPointsToHandle = thrust::minimum<NrPoints>()(
          ThrustSystem<A>::Memory::getFreeBytes(nHostBufferBytes_) /
          (sizeof(NrBoxes) + sizeof(BitPattern<N>) + sizeof(bool) + sizeof(Depth) + sizeof(Flags)) / 4,
          points.second - point_temp_begin);
      
      if (nPointsToHandle == 0) {
        std::cout << "insert: Not enough memory. Insertion not completed." << std::endl;
        break;
      }
      
      // search for superset boxes
      typename ThrustSystem<A>::Vector<NrBoxes> searchVector;
      
      // if depth > this->depth(), a search is not needed
      if (depth == getLeafDepth() || depth <= this->depth()) {
        searchVector.resize(nPointsToHandle);
        this->searchByIterators(thrust::make_pair(point_temp_begin, point_temp_begin + nPointsToHandle),
                                depth, searchVector.begin());
      }
      
      // store result temporarily in a vector since it is later needed and res_begin might be a discard_iterator
      typename ThrustSystem<A>::Vector<bool> isToInsert(nPointsToHandle);
      
      // edit res and set flagsForHit
      if (depth == getLeafDepth()) {
        
        // if search index is getInvalidNrBoxes(), box is inserted if it is not invalid
        thrust::transform_if(typename ThrustSystem<A>::execution_policy(),
                             point_temp_begin, point_temp_begin + nPointsToHandle, // input
                             searchVector.begin(), // stencil
                             isToInsert.begin(), // output
                             thrust::not1(IsInvalidBoxFunctor<N>()), // trafo for input
                             thrust::placeholders::_1 == getInvalidNrBoxes()); // predicate for stencil
        
        auto pi_begin = thrust::make_permutation_iterator(this->begin(), searchVector.begin());
        
        // if search index is not getInvalidNrBoxes(), box is only inserted if it did not hit a box on Depth
        // getMaxDepth() since in this case no leaf could be created
        thrust::transform_if(typename ThrustSystem<A>::execution_policy(),
                             pi_begin, pi_begin + nPointsToHandle, // input
                             searchVector.begin(), // stencil
                             isToInsert.begin(), // output
                             thrust::not1(HasAtLeastDepthFunctor<N>(getMaxDepth<N>())), // trafo for input
                             thrust::placeholders::_1 != getInvalidNrBoxes()); // predicate for stencil
        
        // set flagsForHit
        
        // afterwards searchVector only consists of indices whose corresponding boxes in this
        // that are hit and have at least Depth getMaxDepth()
        thrust::replace_if(typename ThrustSystem<A>::execution_policy(),
                           searchVector.begin(), searchVector.end(), // input
                           isToInsert.begin(), // stencil
                           thrust::identity<bool>(), // predicate for stencil
                           getInvalidNrBoxes()); // value for replacing
        
        thrust::sort(typename ThrustSystem<A>::execution_policy(), searchVector.begin(), searchVector.end());
        auto searchVector_end = thrust::unique(typename ThrustSystem<A>::execution_policy(),
                                               searchVector.begin(), searchVector.end());
        searchVector.resize(searchVector_end - searchVector.begin());
        
        this->setFlagsBySearch(searchVector, flagsForHit, getLeafDepth());
        
      } else if (depth <= this->depth()) {
        
        auto zi2_begin =
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    thrust::make_transform_iterator(searchVector.begin(),
                                                    thrust::placeholders::_1 == getInvalidNrBoxes()),
                    thrust::make_transform_iterator(point_temp_begin, thrust::not1(IsInvalidBoxFunctor<N>()))));
        
        // boxes are inserted if they do not exist in this and if they are valid
        thrust::transform(typename ThrustSystem<A>::execution_policy(),
                          zi2_begin, zi2_begin + nPointsToHandle, // input
                          isToInsert.begin(), // output
                          AllIn2TupleFunctor()); // trafo for input
        
        // set flagsForHit
        
        thrust::sort(typename ThrustSystem<A>::execution_policy(), searchVector.begin(), searchVector.end());
        auto searchVector_end = thrust::unique(typename ThrustSystem<A>::execution_policy(),
                                               searchVector.begin(), searchVector.end());
        searchVector.resize(searchVector_end - searchVector.begin());
        
        // set Flags flagsForHit in the already existing and "hit" boxes
        this->setFlagsBySearch(searchVector, flagsForHit, depth);
        
      } else {
        
        // fill res with true since, in this case, all points but invalid ones must be inserted
        thrust::transform(typename ThrustSystem<A>::execution_policy(),
                          point_temp_begin, point_temp_begin + nPointsToHandle, // input
                          isToInsert.begin(), // output
                          thrust::not1(IsInvalidBoxFunctor<N>())); // input -> bool
        
        // no flagsForHit to set
        
      }
      
      searchVector.clear();
      searchVector.shrink_to_fit();
      
      thrust::copy(typename ThrustSystem<A>::execution_policy(),
                   isToInsert.begin(), isToInsert.end(),
                   res_temp_begin);
      
      // need space for nPointsToHandle for later copy_if
      NrBoxes nNewPointsTemp = _bitPatternVector.size();
      _bitPatternVector.resize(nNewPointsTemp + nPointsToHandle);
      _depthVector.resize(nNewPointsTemp + nPointsToHandle);
      _flagsVector.resize(nNewPointsTemp + nPointsToHandle);
      
      auto ins_box_begin = thrust::make_zip_iterator(thrust::make_tuple(_bitPatternVector.begin(),
                                                                        _depthVector.begin(),
                                                                        _flagsVector.begin()));
      
      // first copy
      thrust::copy(typename ThrustSystem<A>::execution_policy(),
                   point_temp_begin, point_temp_begin + nPointsToHandle,
                   ins_box_begin + nNewPointsTemp);
      
      // then remove if box is found in this / is not supposed to be inserted;
      // in particular, all invalid boxes are removed
      auto ins_box_end = thrust::remove_if(typename ThrustSystem<A>::execution_policy(),
                                           ins_box_begin + nNewPointsTemp, // values_begin
                                           ins_box_begin + nNewPointsTemp + nPointsToHandle, // values_end
                                           isToInsert.begin(), // stencil
                                           thrust::logical_not<bool>()); // predicate for stencil
      
      // using thrust::copy_if instead of thrust::copy and thrust::remove_if causes runtime errors:
      // terminate called after throwing an instance of 'thrust::system::system_error'
      //   what():  function_attributes(): after cudaFuncGetAttributes: invalid device function
      // Abgebrochen (Speicherabzug geschrieben)
      
      isToInsert.clear();
      isToInsert.shrink_to_fit();
      
      // adapt depths of new boxes
      if (depth == getLeafDepth()) {
        // new function for search vector: where would new boxes be inserted in this
        typename ThrustSystem<A>::Vector<NrBoxes> lowerBoundVector(ins_box_end - (ins_box_begin + nNewPointsTemp));
        
        thrust::lower_bound(typename ThrustSystem<A>::execution_policy(),
                            this->begin(), this->end(), // field to search in
                            ins_box_begin + nNewPointsTemp, ins_box_end, // boxes to search
                            lowerBoundVector.begin(), // output: indices
                            IsStrictlyPrecedingFunctor<N>());
        
        // compute new Depths
        auto zi3_begin =
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    // newBox[0], newBox[1], newBox[2], …
                    ins_box_begin + nNewPointsTemp,
                    // Box[max(searchIndex[0]-1,0)], Box[max(searchIndex[1]-1,0)], Box[max(searchIndex[2]-1,0)], …
                    thrust::make_permutation_iterator(
                        this->begin(),
                        thrust::make_transform_iterator(
                            lowerBoundVector.begin(),
                            DecrementWhenPositiveFunctor())),
                    // Box[min(searchIndex[0],end-1)], Box[min(searchIndex[1],end-1)], Box[min(searchIndex[2],end-1)],…
                    thrust::make_permutation_iterator(
                        this->begin(),
                        thrust::make_transform_iterator(
                            lowerBoundVector.begin(),
                            CapByFunctor(this->size() - 1)))));
        
        thrust::transform(typename ThrustSystem<A>::execution_policy(),
                          zi3_begin, zi3_begin + lowerBoundVector.size(), // input
                          ins_box_begin + nNewPointsTemp, // output
                          AdaptDepthOfFirstBoxFunctor<N>()); // change depth
      } else {
        // adapt Depths to depth
        thrust::transform(typename ThrustSystem<A>::execution_policy(),
                          ins_box_begin + nNewPointsTemp, ins_box_end, // input
                          ins_box_begin + nNewPointsTemp, // in-place trafo
                          DecrementDepthToFunctor<N>(depth)); // change Depth to depth
      }
      
      // eliminate multiple new boxes of all already inserted ones
      sortBoxes<A, N>(_bitPatternVector.begin(), _depthVector.begin(), _flagsVector.begin(), ins_box_end - ins_box_begin);
      
      // eliminate multiple new boxes
      ins_box_end = thrust::unique(typename ThrustSystem<A>::execution_policy(),
                                   ins_box_begin, ins_box_end,
                                   IsSameBoxFunctor<N>());
      
      _bitPatternVector.resize(ins_box_end - ins_box_begin);
      _depthVector.resize(ins_box_end - ins_box_begin);
      _flagsVector.resize(ins_box_end - ins_box_begin);
      
      point_temp_begin = point_temp_begin + nPointsToHandle;
      res_temp_begin = res_temp_begin + nPointsToHandle;
    }
    
    // set Flags for inserted boxes
    thrust::transform(typename ThrustSystem<A>::execution_policy(),
                      _flagsVector.begin(), _flagsVector.end(), // values
                      _flagsVector.begin(), // in-place trafo
                      SetFlagsFunctor(flagsForInserted));
    
    NrBoxes nNewPoints = _bitPatternVector.size();
    
    NrBoxes nOldBoxes = this->size();
    this->resize(nOldBoxes + nNewPoints);
    
    auto ins_box_begin = thrust::make_zip_iterator(thrust::make_tuple(_bitPatternVector.begin(),
                                                                      _depthVector.begin(),
                                                                      _flagsVector.begin()));
    
    // copy new boxes now to the end of this
    thrust::copy(typename ThrustSystem<A>::execution_policy(),
                 ins_box_begin, ins_box_begin + nNewPoints,
                 this->begin() + nOldBoxes);
    
    _bitPatternVector.clear();
    _bitPatternVector.shrink_to_fit();
    _depthVector.clear();
    _depthVector.shrink_to_fit();
    _flagsVector.clear();
    _flagsVector.shrink_to_fit();
    
    // sort this, but new boxes with greater depth could be equal in minor depth to previously existing boxes
    sortBoxes<A, N>(bitPatternVector_.begin(), depthVector_.begin(), flagsVector_.begin(), this->size());
    
    // combine "equal" boxes by taking the "deepest" box of the segment,
    // i.e. that box with maximal depth, and combining (or) the flags,
    // only "going" back and forth in the same segment guarantees the "right" thing for all its boxes
    thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(),
                                  this->begin(), this->end(), // keys
                                  this->begin(), // "values" to "sum up"
                                  this->begin(), // in-place trafo
                                  HasSamePathFunctor<N>(), // "segment operator" for keys
                                  ComputeMaxBitPatternMaxDepthOrFlagsFunctor<N>());

    thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(), // Note all reverse_iterators!
                                  this->rbegin(), this->rend(), // keys
                                  this->rbegin(), // "values" to "sum up"
                                  this->rbegin(), // in-place trafo
                                  HasSamePathFunctor<N>(), // "segment operator" for keys
                                  ComputeMaxBitPatternMaxDepthOrFlagsFunctor<N>());
    
    // the "surviving" box of each segment (see inclusive_scan_by_key) has greatest Depth and all Flags of the segment
    auto new_end = thrust::unique(typename ThrustSystem<A>::execution_policy(),
                                  this->begin(), this->end(),
                                  HasSamePathFunctor<N>());
    
    // resize
    this->resize(new_end - this->begin());
    
  } else {
    
    std::cout << "Are you kidding me? How should this work?" << std::endl;
    
  }
  
  this->freeIndicesInDepth();
  this->freeLeafIndicesFromDepth();
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename BoxIteratorPair>
inline typename ThrustSystem<A>::Vector<bool> ImplicitBoxTree<A, DIM, REAL, N>::insertByIterators(
    BoxIteratorPair points, Depth depth, Flags flagsForInserted, Flags flagsForHit)
{
  // return vector
  typename ThrustSystem<A>::Vector<bool> res(points.second - points.first);
  
  this->insertByIterators(points, depth, flagsForInserted, flagsForHit, res.begin());
  
  return res;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename BoolIterator>
inline void ImplicitBoxTree<A, DIM, REAL, N>::insertByPoints(const typename ThrustSystem<A>::Vector<REAL>& points,
                                                             Depth depth, Flags flagsForInserted, Flags flagsForHit,
                                                             BoolIterator res_begin)
{
  auto ti_begin =
      thrust::make_transform_iterator(
          thrust::make_transform_iterator(
              thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                              thrust::placeholders::_1 * NrPoints(DIM)),
              AdvanceConstRealPointerFunctor<REAL>(thrust::raw_pointer_cast(points.data()))),
          ComputeBoxFromPointFunctor<DIM, REAL, N>(thrust::raw_pointer_cast(center_.data()),
                                                   thrust::raw_pointer_cast(radius_.data()),
                                                   thrust::raw_pointer_cast(sdScheme_.data()),
                                                   thrust::raw_pointer_cast(sdCount_.data()),
                                                   this->isSdSchemeDefault(),
                                                   flagsForInserted));
  
  this->insertByIterators(thrust::make_pair(ti_begin, ti_begin + points.size() / DIM),
                          depth, flagsForInserted, flagsForHit, res_begin);
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline typename ThrustSystem<A>::Vector<bool> ImplicitBoxTree<A, DIM, REAL, N>::insertByPoints(
    const typename ThrustSystem<A>::Vector<REAL>& points, Depth depth, Flags flagsForInserted, Flags flagsForHit)
{
  // return vector
  typename ThrustSystem<A>::Vector<bool> res(points.size() / DIM);
  
  this->insertByPoints(points, depth, flagsForInserted, flagsForHit, res.begin());
  
  return res;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline void ImplicitBoxTree<A, DIM, REAL, N>::insertSurroundingBoxIfEmpty()
{
  if (this->size() == 0) {
    this->resize(1);
    thrust::fill(this->begin(), this->end(), getSurroundingBox<N>());
    
    this->freeIndicesInDepth();
    this->freeLeafIndicesFromDepth();
  }
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline bool ImplicitBoxTree<A, DIM, REAL, N>::isSdSchemeDefault() const
{
  auto ti_begin = thrust::make_transform_iterator(thrust::make_counting_iterator(Depth(0)),
                                                  thrust::placeholders::_1 % DIM);
  
  return thrust::equal(typename ThrustSystem<A>::execution_policy(), sdScheme_.begin(), sdScheme_.end(), ti_begin);
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline thrust::transform_iterator<
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
ImplicitBoxTree<A, DIM, REAL, N>::isNewBoxInDepth_begin(Depth depth) const
{
  if (depth != getLeafDepth() && depth != getUnsubdividableDepth() && depth > this->depth()) {
    std::cout << "Warning: Undefined behaviour since desired depth exceeds ImplicitBoxTree's current depth." 
              << std::endl;
  }
  
  // iterator indicating where a new box in decremented Depth depth "starts"
  return thrust::make_transform_iterator(
      thrust::make_zip_iterator(
          thrust::make_tuple(
              // index for box of interest
              thrust::make_counting_iterator(NrBoxes(0)),
              // boxes of interest, i.e. Box[0], Box[1], Box[2], Box[3], Box[4], …
              thrust::make_transform_iterator(this->begin(), DecrementDepthToFunctor<N>(depth)),
              // predecessor of box of interest
              thrust::make_transform_iterator(
                  thrust::make_permutation_iterator( // Box[0], Box[0], Box[1], Box[2], Box[3], ….
                      this->begin(),
                      thrust::make_transform_iterator( // 0, 0, 1, 2, 3, …
                          thrust::make_counting_iterator(NrBoxes(0)),
                          DecrementWhenPositiveFunctor())),
                  DecrementDepthToFunctor<N>(depth)))),
      IsUnequalToPredecessorInDepthFunctor<N>(depth));
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline thrust::transform_iterator<
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
ImplicitBoxTree<A, DIM, REAL, N>::isUnsubdividable_begin(Flags flags) const
{
  // iterator indicating whether a box shall be unsubdivided
  return thrust::make_transform_iterator(
      thrust::make_zip_iterator(
          thrust::make_tuple(
              // boxes to analyse
              this->begin(), //                     Box[0], Box[1], Box[2], …, Box[end-1], Box[end], Box[end+1],…
              // predecessors of boxes
              thrust::make_permutation_iterator( // Box[0], Box[0], Box[1], …, Box[end-2], Box[end-1], Box[end],…
                  this->begin(),
                  thrust::make_transform_iterator(thrust::make_counting_iterator(NrBoxes(0)),
                                                  DecrementWhenPositiveFunctor())),
              // successors of boxes
              thrust::make_permutation_iterator( // Box[1], Box[2], Box[3], …, Box[end-1], Box[end-1], Box[end-1],…
                  this->begin(),
                  thrust::make_transform_iterator(thrust::make_counting_iterator(NrBoxes(1)),
                                                  CapByFunctor(this->size() - NrBoxes(1)))))),
      IsUnsubdividableFunctor<N>(flags));
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<Depth D, typename Grid, typename Map>
inline typename std::enable_if<
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
ImplicitBoxTree<A, DIM, REAL, N>::makePointIteratorPair(Grid grid, Map map, Flags flagsToSet)
{
  return this->makePointIteratorPairOverRoot(grid, map, flagsToSet);
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename Grid, typename Map>
inline thrust::pair<
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
ImplicitBoxTree<A, DIM, REAL, N>::makePointIteratorPairOverRoot(Grid grid, Map map, Flags flagsToSet)
{
  auto it =
      thrust::make_zip_iterator(
          thrust::make_tuple(
              thrust::make_counting_iterator(NrPoints(0)), // input sequence for grid()
              thrust::make_constant_iterator(getSurroundingBox<N>()))); // boxes
  
  auto res_begin =
      thrust::make_transform_iterator(
          it,
          GridPointsFunctor<DIM, REAL, N, Grid, Map>(grid,
                                                     map,
                                                     flagsToSet,
                                                     thrust::raw_pointer_cast(center_.data()),
                                                     thrust::raw_pointer_cast(radius_.data()),
                                                     thrust::raw_pointer_cast(sdScheme_.data()),
                                                     thrust::raw_pointer_cast(sdCount_.data()),
                                                     this->isSdSchemeDefault()));
  
  auto res_end = res_begin + grid.getNumberOfPointsPerBox();
  
  return thrust::make_pair(res_begin, res_end);
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<Depth D, typename Grid, typename Map>
inline typename std::enable_if<
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
ImplicitBoxTree<A, DIM, REAL, N>::makePointIteratorPair(Grid grid, Map map, Flags flagsToSet)
{
  return this->makePointIteratorPairOverLeaves(grid, map, flagsToSet);
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename Grid, typename Map>
inline thrust::pair<
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
ImplicitBoxTree<A, DIM, REAL, N>::makePointIteratorPairOverLeaves(Grid grid, Map map, Flags flagsToSet)
{
  // "grid.getNumberOfPointsPerBox() times repeat iterator", i.e. Box[0], …, Box[0], Box[1], …, Box[1], Box[2], …
  auto repeat_box_begin = thrust::make_permutation_iterator(this->begin(),
                                                            thrust::make_transform_iterator(
                                                                thrust::make_counting_iterator(NrPoints(0)),
                                                                IDivideByFunctor(grid.getNumberOfPointsPerBox())));
  
  auto it =
      thrust::make_zip_iterator(
          thrust::make_tuple(
              thrust::make_counting_iterator(NrPoints(0)), // input sequence for grid()
              repeat_box_begin)); // boxes
  
  auto res_begin =
      thrust::make_transform_iterator(
          it,
          GridPointsFunctor<DIM, REAL, N, Grid, Map>(grid,
                                                     map,
                                                     flagsToSet,
                                                     thrust::raw_pointer_cast(center_.data()),
                                                     thrust::raw_pointer_cast(radius_.data()),
                                                     thrust::raw_pointer_cast(sdScheme_.data()),
                                                     thrust::raw_pointer_cast(sdCount_.data()),
                                                     this->isSdSchemeDefault()));
  
  auto res_end = res_begin + this->count(getLeafDepth()) * grid.getNumberOfPointsPerBox();
  
  return thrust::make_pair(res_begin, res_end);
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<Depth D, typename Grid, typename Map>
inline typename std::enable_if<
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
ImplicitBoxTree<A, DIM, REAL, N>::makePointIteratorPair(Grid grid, Map map, Flags flagsToSet)
{
  return this->makePointIteratorPairOverDepth(D, grid, map, flagsToSet);
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename Grid, typename Map>
inline thrust::pair<
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
ImplicitBoxTree<A, DIM, REAL, N>::makePointIteratorPairOverDepth(Depth depth, Grid grid, Map map, Flags flagsToSet)
{
  return this->makePointIteratorPairByFlags(depth, Flag::NONE, grid, map, flagsToSet);
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename Grid, typename Map>
inline thrust::pair<
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
ImplicitBoxTree<A, DIM, REAL, N>::makePointIteratorPairByFlags(Depth depth, Flags flagsToCheck,
                                                               Grid grid, Map map, Flags flagsToSet)
{
  this->initializeLeafIndicesFromDepth(depth, flagsToCheck);
  
  // "grid.getNumberOfPointsPerBox() times repeat iterator",
  // i.e. leafIndicesFromDepth[0], …, leafIndicesFromDepth[0], leafIndicesFromDepth[1], …, 
  // leafIndicesFromDepth[1], leafIndicesFromDepth[2], …
  auto repeat_ind_begin =
      thrust::make_permutation_iterator(
          leafIndicesFromDepth_.begin(),
          thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                          IDivideByFunctor(grid.getNumberOfPointsPerBox())));
  
  // "grid.getNumberOfPointsPerBox() times repeat iterator", i.e. Box[0], …, Box[0], Box[1], …, Box[1], Box[2], …
  auto repeat_box_begin = thrust::make_permutation_iterator(this->begin(), repeat_ind_begin);
  
  auto it =
      thrust::make_zip_iterator(
          thrust::make_tuple(
              thrust::make_counting_iterator(NrPoints(0)), // input sequence for grid()
              thrust::make_transform_iterator(repeat_box_begin, DecrementDepthToFunctor<N>(depth)))); // boxes
  
  auto res_begin =
      thrust::make_transform_iterator(
          it,
          GridPointsFunctor<DIM, REAL, N, Grid, Map>(grid,
                                                     map,
                                                     flagsToSet,
                                                     thrust::raw_pointer_cast(center_.data()),
                                                     thrust::raw_pointer_cast(radius_.data()),
                                                     thrust::raw_pointer_cast(sdScheme_.data()),
                                                     thrust::raw_pointer_cast(sdCount_.data()),
                                                     this->isSdSchemeDefault()));
  
  auto res_end = res_begin + leafIndicesFromDepth_.size() * grid.getNumberOfPointsPerBox();
  
  return thrust::make_pair(res_begin, res_end);
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline NrBoxes ImplicitBoxTree<A, DIM, REAL, N>::remove(Flags flags)
{
  NrBoxes nLeaves = this->count(getLeafDepth());
  
  // temporary vector since the stencil in remove_if shall not overlap the input
  typename ThrustSystem<A>::Vector<Flags> _flagsVector(flagsVector_.begin(), flagsVector_.end());
  
  // box_new_end points to the end of the sequence whose boxes have all Flags flags set
  auto box_new_end = thrust::remove_if(typename ThrustSystem<A>::execution_policy(),
                                       this->begin(), this->end(), // values to perhaps remove
                                       _flagsVector.begin(), // stencil
                                       thrust::not1(AreFlagsSetFunctor(flags))); // predicate for stencil
  
  // resize ImplicitBoxTree
  this->resize(box_new_end - this->begin());
  
  NrBoxes res = nLeaves - this->count(getLeafDepth());
  
  // have at least the "root" in the "tree"
  this->insertSurroundingBoxIfEmpty();
  
  if (res > 0) {
    this->freeIndicesInDepth();
    this->freeLeafIndicesFromDepth();
  }
  
  return res;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename BoxIteratorPair, typename NrBoxesIterator>
inline void ImplicitBoxTree<A, DIM, REAL, N>::searchByIterators(BoxIteratorPair points,
                                                                Depth depth,
                                                                NrBoxesIterator res_begin)
{
  NrPoints nPoints = points.second - points.first;
  auto res_end = res_begin + nPoints;
  
  NrBoxes nBoxes = this->size();
  
  if (depth == 0) {
    
    thrust::fill(typename ThrustSystem<A>::execution_policy(), res_begin, res_end, NrBoxes(0));
    
    thrust::replace_if(typename ThrustSystem<A>::execution_policy(),
                       res_begin, res_end,
                       points.first, // stencil
                       IsInvalidBoxFunctor<N>(), // predicate for stencil
                       getInvalidNrBoxes());
    
  } else if (depth == getLeafDepth() || this->areAllLeavesOnDepth(depth)) {
    
    // compute "lower-bound-indices"
    thrust::lower_bound(typename ThrustSystem<A>::execution_policy(),
                        this->begin(), this->end(), // find box indices
                        points.first, points.second, // points to search
                        res_begin, // output: indices in Depth getLeafDepth()
                        IsStrictlyPrecedingFunctor<N>());
    
    // tuple iterator: <searchIndex,
    //                  Point,
    //                  Box[searchIndex-1],
    //                  Box[searchIndex],
    //                  true,
    //                  true>
    auto zi_begin =
        thrust::make_zip_iterator(
            thrust::make_tuple(
                // search[0], search[1], search[2], …
                res_begin,
                // Point[0], Point[1], Point[2], Point[3], …
                points.first,
                // Box[max(search[0]-1,0)], Box[max(search[1]-1,0)], Box[max(search[2]-1,0)], …
                thrust::make_permutation_iterator(
                    this->begin(),
                    thrust::make_transform_iterator(res_begin, DecrementWhenPositiveFunctor())),
                // Box[min(search[0],size-1)], Box[min(search[1],size-1)], Box[min(search[2],size-1)], …
                thrust::make_permutation_iterator(
                    this->begin(),
                    thrust::make_transform_iterator(res_begin, CapByFunctor(nBoxes - NrBoxes(1)))),
                // two times true since each box can be considered
                thrust::make_constant_iterator(true),
                thrust::make_constant_iterator(true)));
    
    // store search indeces in res_begin
    thrust::transform(typename ThrustSystem<A>::execution_policy(),
                      zi_begin, zi_begin + nPoints,
                      res_begin,
                      ComputeSearchIndexFunctor<N>());
    
  } else if (depth == getUnsubdividableDepth()) {
    
    auto it = this->isUnsubdividable_begin();
    typename ThrustSystem<A>::Vector<bool> isUnsubdividableVector(it, it + nBoxes);
    
    // compute "lower-bound-indices"
    thrust::lower_bound(typename ThrustSystem<A>::execution_policy(),
                        this->begin(), this->end(), // find box indices
                        points.first, points.second, // points to search
                        res_begin, // output: indices in Depth getLeafDepth()
                        IsStrictlyPrecedingFunctor<N>());
    
    // tuple iterator: <searchIndex in Depth getLeafDepth(),
    //                  Point, 
    //                  Box[searchIndex-1] in Depth depth,
    //                  Box[searchIndex] in Depth depth,
    //                  isUnsubdividable_begin[searchIndex-1],
    //                  isUnsubdividable_begin[searchIndex]>
    auto zi_begin =
        thrust::make_zip_iterator(
            thrust::make_tuple(
                // search[0], search[1], search[2], …
                res_begin,
                // Point[0], Point[1], Point[2], Point[3], …
                points.first,
                thrust::make_transform_iterator(
                    // Box[max(search[0]-1,0)], Box[max(search[1]-1,0)], Box[max(search[2]-1,0)], …
                    thrust::make_permutation_iterator(
                        this->begin(),
                        thrust::make_transform_iterator(res_begin, DecrementWhenPositiveFunctor())),
                    DecrementDepthByFunctor<N>(1)),
                thrust::make_transform_iterator(
                    // Box[min(search[0],size-1)], Box[min(search[1],size-1)], Box[min(search[2],size-1)], …
                    thrust::make_permutation_iterator(
                        this->begin(),
                        thrust::make_transform_iterator(res_begin, CapByFunctor(nBoxes - NrBoxes(1)))),
                    DecrementDepthByFunctor<N>(1)),
                // indicates whether the first box of this in here is unsubdividable
                thrust::make_permutation_iterator(
                    isUnsubdividableVector.begin(),
                    thrust::make_transform_iterator(res_begin, DecrementWhenPositiveFunctor())),
                // indicates whether the second box of this in here is unsubdividable
                thrust::make_permutation_iterator(
                    isUnsubdividableVector.begin(),
                    thrust::make_transform_iterator(res_begin, CapByFunctor(nBoxes - NrBoxes(1))))));
    
    // already transform so that points.first … points.second are not used anymore;
    // see also comment before next init_indices_in_decremented_depth
    thrust::transform(typename ThrustSystem<A>::execution_policy(),
                      zi_begin, zi_begin + nPoints,
                      res_begin,
                      ComputeSearchIndexFunctor<N>());
    
    // initialise boxes' indeces in decremented Depth if needed
    this->initializeIndicesInDepth(getUnsubdividableDepth());
    
    // pi_begin[i] gives the index in decremented Depth of point points.first[i]
    auto pi_begin = thrust::make_permutation_iterator(indicesInDepth_.begin(), res_begin);
    
    // store index in decremented Depth in res if index' box really exist in this
    thrust::replace_copy_if(typename ThrustSystem<A>::execution_policy(),
                            pi_begin, pi_begin + nPoints, // indices in decremented Depth to eventually copy
                            res_begin, // stencil
                            res_begin, // output
                            thrust::placeholders::_1 == getInvalidNrBoxes(), // predicate for stencil
                            getInvalidNrBoxes()); // value if index is "wrong" (since lower_bound != "searchIndex")
    
  } else if (depth <= this->depth()) {
    
    auto it = thrust::make_transform_iterator(this->begin(), HasAtLeastDepthFunctor<N>(depth));
    typename ThrustSystem<A>::Vector<bool> hasAtLeastDepthVector(it, it + nBoxes);
    
    // compute "lower-bound-indices"
    thrust::lower_bound(typename ThrustSystem<A>::execution_policy(),
                        this->begin(), this->end(), // find box indices
                        points.first, points.second, // points to search
                        res_begin, // output: indices in Depth getLeafDepth()
                        IsStrictlyPrecedingFunctor<N>());
    
    // tuple iterator: <searchIndex in Depth getLeafDepth(),
    //                  Point, 
    //                  Box[searchIndex-1] in Depth depth,
    //                  Box[searchIndex] in Depth depth,
    //                  hasAtLeastDepth[searchIndex-1],
    //                  hasAtLeastDepth[searchIndex]>
    auto zi_begin =
        thrust::make_zip_iterator(
            thrust::make_tuple(
                // search[0], search[1], search[2], …
                res_begin,
                // Point[0], Point[1], Point[2], Point[3], …
                points.first,
                thrust::make_transform_iterator(
                    // Box[max(search[0]-1,0)], Box[max(search[1]-1,0)], Box[max(search[2]-1,0)], …
                    thrust::make_permutation_iterator(
                        this->begin(),
                        thrust::make_transform_iterator(res_begin, DecrementWhenPositiveFunctor())),
                    DecrementDepthToFunctor<N>(depth)),
                thrust::make_transform_iterator(
                    // Box[min(search[0],size-1)], Box[min(search[1],size-1)], Box[min(search[2],size-1)], …
                    thrust::make_permutation_iterator( 
                        this->begin(),
                        thrust::make_transform_iterator(res_begin, CapByFunctor(nBoxes - NrBoxes(1)))),
                    DecrementDepthToFunctor<N>(depth)),
                // indicates whether the first box of this in here is unsubdividable
                thrust::make_permutation_iterator(
                    hasAtLeastDepthVector.begin(),
                    thrust::make_transform_iterator(res_begin, DecrementWhenPositiveFunctor())),
                // indicates whether the second box of this in here is unsubdividable
                thrust::make_permutation_iterator(
                    hasAtLeastDepthVector.begin(),
                    thrust::make_transform_iterator(res_begin, CapByFunctor(nBoxes - NrBoxes(1))))));
    
    // already transform so that points.first … points.second are not used anymore;
    // see also comment before next init_indices_in_decremented_depth
    thrust::transform(typename ThrustSystem<A>::execution_policy(),
                      zi_begin, zi_begin + nPoints,
                      res_begin,
                      ComputeSearchIndexFunctor<N>());
    
    // initialise boxes' indeces in decremented Depth if needed
    this->initializeIndicesInDepth(depth);
    
    // pi_begin[i] gives the index in decremented Depth of point points.first[i]
    auto pi_begin = thrust::make_permutation_iterator(indicesInDepth_.begin(), res_begin);
    
    // store index in decremented Depth in res if index' box really exist in this
    thrust::replace_copy_if(typename ThrustSystem<A>::execution_policy(),
                            pi_begin, pi_begin + nPoints, // indices in decremented Depth to eventually copy
                            res_begin, // stencil
                            res_begin, // output
                            thrust::placeholders::_1 == getInvalidNrBoxes(), // predicate for stencil
                            getInvalidNrBoxes()); // value if index is "wrong" (since lower_bound != "searchIndex")
    
  } else {
    
    thrust::fill(typename ThrustSystem<A>::execution_policy(), res_begin, res_end, getInvalidNrBoxes());
    
  }
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename BoxIteratorPair>
inline typename ThrustSystem<A>::Vector<NrBoxes> ImplicitBoxTree<A, DIM, REAL, N>::searchByIterators(
    BoxIteratorPair points, Depth depth)
{
  // return vector
  typename ThrustSystem<A>::Vector<NrBoxes> res(points.second - points.first);
  
  this->searchByIterators(points, depth, res.begin());
  
  return res;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename NrBoxesIterator>
void ImplicitBoxTree<A, DIM, REAL, N>::searchByPoints(
    const typename ThrustSystem<A>::Vector<REAL>& points, Depth depth, NrBoxesIterator res_begin)
{
  auto ti_begin =
      thrust::make_transform_iterator(  
          thrust::make_transform_iterator(
              thrust::make_transform_iterator(
                  thrust::make_counting_iterator(NrPoints(0)),
                  thrust::placeholders::_1 * NrPoints(DIM)),
              AdvanceConstRealPointerFunctor<REAL>(thrust::raw_pointer_cast(points.data()))),
          ComputeBoxFromPointFunctor<DIM, REAL, N>(thrust::raw_pointer_cast(center_.data()), 
                                                   thrust::raw_pointer_cast(radius_.data()), 
                                                   thrust::raw_pointer_cast(sdScheme_.data()),
                                                   thrust::raw_pointer_cast(sdCount_.data()),
                                                   this->isSdSchemeDefault(),
                                                   Flag::NONE));
  
  this->searchByIterators(thrust::make_pair(ti_begin, ti_begin + points.size() / DIM), depth, res_begin);
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline typename ThrustSystem<A>::Vector<NrBoxes> ImplicitBoxTree<A, DIM, REAL, N>::searchByPoints(
    const typename ThrustSystem<A>::Vector<REAL>& points, Depth depth)
{
  // return vector
  typename ThrustSystem<A>::Vector<NrBoxes> res(points.size() / DIM);
  
  this->searchByPoints(points, depth, res.begin());
  
  return res;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename BoxIteratorPair>
inline typename ThrustSystem<A>::Vector<NrBoxes> ImplicitBoxTree<A, DIM, REAL, N>::searchCompacted(
    BoxIteratorPair points, Depth depth)
{
  // return vector
  typename ThrustSystem<A>::Vector<NrBoxes> res;
  
  if (depth == 0) {
    
    bool areAllBoxesInvalid = thrust::all_of(typename ThrustSystem<A>::execution_policy(),
                                             points.first, points.second,
                                             IsInvalidBoxFunctor<N>());
    
    if (!areAllBoxesInvalid) {
      res.resize(1);
      res.front() = 0;
    }
    
  } else if (depth == getLeafDepth() || depth == getUnsubdividableDepth() || depth <= this->depth()) {
    
    // a quarter to "ensure" enough space for later search, sort, and unique
    res.resize(ThrustSystem<A>::Memory::getFreeBytes(nHostBufferBytes_) / sizeof(NrBoxes) / 4);
    auto res_temp_begin = res.begin();
    auto point_temp_begin = points.first;
    
    NrBoxes nBoxes = this->count(depth);
    
    while (points.second - point_temp_begin > 0  &&  res_temp_begin - res.begin() < nBoxes) {
      
      NrPoints nPointsToHandle = thrust::minimum<NrPoints>()(res.end() - res_temp_begin,
                                                             points.second - point_temp_begin);
      
      if (nPointsToHandle == 0) {
        std::cout << "searchCompacted: Not enough memory. Search not completed." << std::endl;
        break;
      }
      
      this->searchByIterators(thrust::make_pair(point_temp_begin, point_temp_begin + nPointsToHandle),
                              depth,
                              res_temp_begin);
      
      // sort all "hit" indices
      thrust::sort(typename ThrustSystem<A>::execution_policy(), res.begin(), res_temp_begin + nPointsToHandle);
      // make range unique
      res_temp_begin = thrust::unique(typename ThrustSystem<A>::execution_policy(),
                                      res.begin(), res_temp_begin + nPointsToHandle);
      // eliminate getInvalidNrBoxes() (which stands at the end) if it exists:
      res_temp_begin = res_temp_begin - NrBoxes(*(res_temp_begin - 1) == getInvalidNrBoxes());
      
      point_temp_begin = point_temp_begin + nPointsToHandle;
    }
    
    // resize res
    res.resize(res_temp_begin - res.begin());
    res.shrink_to_fit();
  }
  
  return res;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline NrBoxes ImplicitBoxTree<A, DIM, REAL, N>::setFlags(Flags flags, Depth depth)
{
  return this->setFlagsByStencil(thrust::make_constant_iterator(true), flags, depth);
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename BoxIteratorPair>
inline NrBoxes ImplicitBoxTree<A, DIM, REAL, N>::setFlagsByIterators(BoxIteratorPair points, Flags flags, Depth depth)
{
  return this->setFlagsBySearch(this->searchCompacted(points, depth), flags, depth);
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline NrBoxes ImplicitBoxTree<A, DIM, REAL, N>::setFlagsByPoints(const typename ThrustSystem<A>::Vector<REAL>& points,
                                                                  Flags flags, Depth depth)
{
  auto ti_begin =
      thrust::make_transform_iterator(
          thrust::make_transform_iterator(
              thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                              thrust::placeholders::_1 * NrPoints(DIM)),
              AdvanceConstRealPointerFunctor<REAL>(thrust::raw_pointer_cast(points.data()))),
          ComputeBoxFromPointFunctor<DIM, REAL, N>(thrust::raw_pointer_cast(center_.data()), 
                                                   thrust::raw_pointer_cast(radius_.data()), 
                                                   thrust::raw_pointer_cast(sdScheme_.data()),
                                                   thrust::raw_pointer_cast(sdCount_.data()),
                                                   this->isSdSchemeDefault(),
                                                   Flag::NONE));
  
  return this->setFlagsByIterators(thrust::make_pair(ti_begin, ti_begin + points.size() / DIM), flags, depth);
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline NrBoxes ImplicitBoxTree<A, DIM, REAL, N>::setFlagsBySearch(
    const typename ThrustSystem<A>::Vector<NrBoxes>& searchVector, Flags flags, Depth depth)
{
  // if last element is getInvalidNrBoxes(), then reduce later the end()-iterator
  NrBoxes isLastElementInvalid = searchVector.size() ? searchVector.back() == getInvalidNrBoxes() : 0;
    
  if (depth == 0) {
    
    bool isSurroundingBoxHit = thrust::any_of(typename ThrustSystem<A>::execution_policy(),
                                              searchVector.begin(), searchVector.end() - isLastElementInvalid,
                                              thrust::placeholders::_1 == NrBoxes(0));
    
    bool haveAllBoxesFlagsSet = thrust::all_of(typename ThrustSystem<A>::execution_policy(),
                                               flagsVector_.begin(), flagsVector_.end(),
                                               AreFlagsSetFunctor(flags));
    
    if (isSurroundingBoxHit && !haveAllBoxesFlagsSet) {
      
      thrust::transform(typename ThrustSystem<A>::execution_policy(),
                        flagsVector_.begin(), flagsVector_.end(),
                        flagsVector_.begin(),
                        SetFlagsFunctor(flags));
      
      if (isAnyFlagSet(flagsToCheckForLeafIndices_, flags)) {
        this->freeLeafIndicesFromDepth();
      }
      
      return 1;
      
    } else {
      
      return 0;
      
    }
    
  } else if (depth == getLeafDepth() || this->areAllLeavesOnDepth(depth)) {
    
    NrBoxes res = thrust::count_if(typename ThrustSystem<A>::execution_policy(),
                                   thrust::make_permutation_iterator(flagsVector_.begin(),
                                                                     searchVector.begin()), // begin
                                   thrust::make_permutation_iterator(flagsVector_.begin(),
                                                                     searchVector.end() - isLastElementInvalid), // end
                                   thrust::not1(AreFlagsSetFunctor(flags))); // count boxes that have flags not set
    
    thrust::transform_if(typename ThrustSystem<A>::execution_policy(),
                         thrust::make_permutation_iterator(flagsVector_.begin(), searchVector.begin()), // begin
                         thrust::make_permutation_iterator(flagsVector_.begin(),
                                                           searchVector.end() - isLastElementInvalid), // end
                         searchVector.begin(), // stencil
                         thrust::make_permutation_iterator(flagsVector_.begin(),
                                                           searchVector.begin()), // in-place trafo
                         SetFlagsFunctor(flags), // operation to execute
                         thrust::placeholders::_1 != getInvalidNrBoxes()); // predicate for stencil
    
    if (res > 0 && isAnyFlagSet(flagsToCheckForLeafIndices_, flags)) {
      this->freeLeafIndicesFromDepth();
    }
    
    return res;
    
  } else if (depth == getUnsubdividableDepth() || depth <= this->depth()) {
    
    // fill stencilVector with "false"
    typename ThrustSystem<A>::Vector<bool> stencilVector(this->count(depth), false);
    
    thrust::transform_if(typename ThrustSystem<A>::execution_policy(),
                         thrust::make_permutation_iterator(stencilVector.begin(), searchVector.begin()), // begin
                         thrust::make_permutation_iterator(stencilVector.begin(),
                                                           searchVector.end() - isLastElementInvalid), // end
                         searchVector.begin(), // stencil for trafo
                         thrust::make_permutation_iterator(stencilVector.begin(),
                                                           searchVector.begin()), // in-place trafo
                         thrust::logical_not<bool>(), // stencilVector (false) -> not(stencilVector) (true)
                         thrust::placeholders::_1 != getInvalidNrBoxes()); // predicate for stencil
    
    NrBoxes res = this->setFlagsByStencil(stencilVector.begin(), flags, depth);
    
    return res;
    
  } else {
    
    return 0;
    
  }
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename BoolIterator>
inline NrBoxes ImplicitBoxTree<A, DIM, REAL, N>::setFlagsByStencil(BoolIterator stencil_begin,
                                                                   Flags flags, Depth depth)
{
  NrBoxes nBoxes = this->countIfFlags(thrust::not1(AreFlagsSetFunctor(flags)), FlagsAndFunctor(),
                                      stencil_begin, depth);
  
  if (depth == 0) {
    
    if (*stencil_begin) { // must stand inside then-branch!
      thrust::transform(typename ThrustSystem<A>::execution_policy(),
                        flagsVector_.begin(), flagsVector_.end(),
                        flagsVector_.begin(),
                        SetFlagsFunctor(flags));
    }
    
  } else if (depth == getLeafDepth() || this->areAllLeavesOnDepth(depth)) {
    
    thrust::transform_if(typename ThrustSystem<A>::execution_policy(),
                         flagsVector_.begin(), flagsVector_.end(),
                         stencil_begin, // stencil
                         flagsVector_.begin(), // in-place trafo
                         SetFlagsFunctor(flags),
                         thrust::identity<bool>()); // predicate for stencil
    
  } else if (depth == getUnsubdividableDepth()) {
    
    // initialise boxes' indeces in decremented Depth if needed
    this->initializeIndicesInDepth(getUnsubdividableDepth());
    
    // pi_begin[i] == stencil_begin[indicesInDepth_[i]]
    auto pi_begin = thrust::make_permutation_iterator(stencil_begin, indicesInDepth_.begin());
    
    // store values since stencil and in-/output range would otherwise overlap in later transform_if
    typename ThrustSystem<A>::Vector<bool> isUnsubdividableVector(this->isUnsubdividable_begin(),
                                                                  this->isUnsubdividable_begin() + this->size());
    
    // fusing stencil and isUnsubdividable
    auto zi_begin = thrust::make_zip_iterator(thrust::make_tuple(pi_begin,
                                                                 isUnsubdividableVector.begin()));
    
    // set Flags flags in boxes b if b is unsubdividable and stencil_begin[indicesInDepth_[*b]] is true
    thrust::transform_if(typename ThrustSystem<A>::execution_policy(),
                         flagsVector_.begin(), flagsVector_.end(),
                         zi_begin, // stencil
                         flagsVector_.begin(), // in-place trafo
                         SetFlagsFunctor(flags),
                         AllIn2TupleFunctor()); // logical and in tuple of stencil; predicate for stencil
    
  } else if (depth <= this->depth()) {
    
    // initialise boxes' indeces in decremented Depth if needed
    this->initializeIndicesInDepth(depth);
    
    // pi_begin[i] == stencil_begin[indicesInDepth_[i]]
    auto pi_begin = thrust::make_permutation_iterator(stencil_begin, indicesInDepth_.begin());
    
    // depth_begin[i] == hasAtLeastDepth(this[i])
    auto depth_begin = thrust::make_transform_iterator(this->begin(), HasAtLeastDepthFunctor<N>(depth));
    
    // fusing stencil and hasAtLeastDepth
    auto zi_begin = thrust::make_zip_iterator(thrust::make_tuple(pi_begin, depth_begin));
    
    // set Flags flags in boxes b if b has at least Depth depth and stencil_begin[indicesInDepth_[*b]] is true
    thrust::transform_if(typename ThrustSystem<A>::execution_policy(),
                         flagsVector_.begin(), flagsVector_.end(),
                         zi_begin, // stencil
                         flagsVector_.begin(), // in-place trafo
                         SetFlagsFunctor(flags),
                         AllIn2TupleFunctor()); // logical and in tuple of stencil; predicate for stencil
  }
  
  if (nBoxes > 0 && isAnyFlagSet(flagsToCheckForLeafIndices_, flags)) {
    this->freeLeafIndicesFromDepth();
  }
  
  return nBoxes;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline NrBoxes ImplicitBoxTree<A, DIM, REAL, N>::subdivide(Flags flags)
{
  if (this->depth() == getMaxDepth<N>()) {
    std::cout << "Warning: Maximal depth reached for some boxes. Some boxes might therefore be not subdivided."
              << std::endl;
  }
  
  // number of boxes whose Flags flags are set
  NrBoxes res = this->countIfFlags(AreFlagsSetFunctor(flags), FlagsAndFunctor(), getLeafDepth());
  
  // first allocate memory for new boxes at the end so that if it fails no boxes are changed
  NrBoxes nOldBoxes = this->size();
  this->resize(2 * nOldBoxes);
  
  // copies subdividable boxes to the end; nothing is removed
  auto new_end = thrust::copy_if(typename ThrustSystem<A>::execution_policy(),
                                 this->begin(), this->begin() + nOldBoxes,
                                 this->begin() + nOldBoxes,
                                 IsSubdividableFunctor<N>(flags));
  
  this->resize(new_end - this->begin());
  
  // increment depth of appropriate boxes, i.e. make box its 'left' half;
  // first incrementing Depth and then copying leads to wrong results in Depth getMaxDepth<N>() since then the boxes
  // are not subdividable anymore
  thrust::transform_if(typename ThrustSystem<A>::execution_policy(),
                       this->begin(), this->end(),
                       this->begin(), // in-place trafo
                       IncrementDepthByFunctor<N>(1),
                       IsSubdividableFunctor<N>(flags)); // predicate
  
  // make new ('left'-half-)boxes to the 'right' half
  thrust::transform(typename ThrustSystem<A>::execution_policy(),
                    this->begin() + nOldBoxes, this->end(),
                    this->begin() + nOldBoxes,
                    ComputeOtherHalfFunctor<N>());
  
  // sorting
  sortBoxes<A, N>(bitPatternVector_.begin(), depthVector_.begin(), flagsVector_.begin(), this->size());
  
  if (res > 0) {
    this->freeIndicesInDepth();
    this->freeLeafIndicesFromDepth();
  }
  
  return res;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename T, typename BoxIteratorPair>
inline void ImplicitBoxTree<A, DIM, REAL, N>::transitionMatrix(BoxIteratorPair points,
                                                               Depth depth, NrPoints nPointsPerBox,
                                                               bool useAbsoluteValues,
                                                               bool useRowMajor,
                                                               bool useInvalidNrBoxes,
                                                               CooMatrix<A, T>& res)
{
  // use a third of the remaining memory otherwise "out of memory" is more likely
  uint64_t nre = ThrustSystem<A>::Memory::getFreeBytes(nHostBufferBytes_) /
                 (2 * sizeof(NrBoxes) + sizeof(NrPoints)) / 3;
  
  // create matrix with nre elements
  res.resize(nre);
  // replace values-vector with an integral-type vector for better efficiency and performance
  res.values_.clear();
  res.values_.shrink_to_fit();
  typename ThrustSystem<A>::Vector<NrPoints> valuesVector(nre);
  
  // create help iterators for iterating/storing/etc.
  auto row_inds_temp_begin = res.rowIndices_.begin();
  auto col_inds_temp_begin = res.columnIndices_.begin();
  auto index_begin = thrust::make_zip_iterator(thrust::make_tuple(res.rowIndices_.begin(),
                                                                  res.columnIndices_.begin()));
  auto index_temp_begin = index_begin;
  auto values_begin = valuesVector.begin();
  auto values_temp_begin = values_begin;
  auto point_temp_begin = points.first;
  
  // box where each point is mapped from ("start box")
  // Note: input must be NrPoints while output can be converted to NrBoxes
  auto delay_begin = thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                                     thrust::placeholders::_1 / nPointsPerBox);
  
  // iterate until no more points are considered
  while (points.second - point_temp_begin > 0) {
    
    // only use as many points as possible (w.r.t. memory) and still left
    NrPoints nPointsToHandle = thrust::minimum<NrPoints>()(valuesVector.end() - values_temp_begin,
                                                           points.second - point_temp_begin);
    
    // if valuesVector's end is reached, try to allocate more memory
    if (nPointsToHandle == 0) {
      NrPoints vecSize = valuesVector.size();
      NrPoints diff = values_temp_begin - values_begin;
      
      // use a quarter of the remaining memory + the matrix entries in the current column because
      // these elements still need to be sorted and made unique
      uint64_t addSize = (ThrustSystem<A>::Memory::getFreeBytes(nHostBufferBytes_) / 
                         (2 * sizeof(NrBoxes) + sizeof(NrPoints)) + diff) / 4;
      
      if (addSize <= diff) {
        // here, the matrix would even be truncated
        std::cout << "transitionMatrix: Not enough memory. Computation not completed." << std::endl;
        break;
      } else {
        // try to allocate additional memory
        // but if the remaining free memory is pretty small, the allocation might also fail
        res.rowIndices_.resize(vecSize - diff + addSize);
        res.columnIndices_.resize(vecSize - diff + addSize);
        valuesVector.resize(vecSize - diff + addSize);
        
        row_inds_temp_begin = res.rowIndices_.begin() + vecSize;
        col_inds_temp_begin = res.columnIndices_.begin() + vecSize;
        index_temp_begin = thrust::make_zip_iterator(thrust::make_tuple(res.rowIndices_.begin(),
                                                                        res.columnIndices_.begin())) + vecSize;
        index_begin = index_temp_begin - diff;
        values_temp_begin = valuesVector.begin() + vecSize;
        values_begin = values_temp_begin - diff;
        continue;
      }
    }
    
    // get search indices of next mapped points …
    this->searchByIterators(thrust::make_pair(point_temp_begin, point_temp_begin + nPointsToHandle), 
                            depth, col_inds_temp_begin);
    
    // get "start box" indices of next mapped points
    thrust::copy(typename ThrustSystem<A>::execution_policy(),
                 delay_begin + (point_temp_begin - points.first),
                 delay_begin + (point_temp_begin - points.first) + nPointsToHandle,
                 row_inds_temp_begin);
    
    // each new point has quantity 1 for the moment
    thrust::fill(typename ThrustSystem<A>::execution_policy(),
                 values_temp_begin, values_temp_begin + nPointsToHandle,
                 NrPoints(1));
    
    // sort new and existing entries
    sortRowMajorByIterators<A>(thrust::get<0>(index_begin.get_iterator_tuple()),
                               thrust::get<1>(index_begin.get_iterator_tuple()),
                               values_begin,
                               (index_temp_begin + nPointsToHandle) - index_begin);
    
    // first element of segment, i.e. same index, gets sum
    thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(),
                                  thrust::make_reverse_iterator(index_temp_begin + nPointsToHandle), // keys_rbegin
                                  thrust::make_reverse_iterator(index_begin), // keys_rend
                                  thrust::make_reverse_iterator(values_temp_begin + nPointsToHandle), // values_rbegin
                                  thrust::make_reverse_iterator(values_temp_begin + nPointsToHandle)); // output_rbegin
    
    // only keep first element of segment
    auto itPair = thrust::unique_by_key(typename ThrustSystem<A>::execution_policy(),
                                        index_begin, // keys_begin
                                        index_temp_begin + nPointsToHandle, // keys_end
                                        values_begin);
    
    // update iterators where new entries can be stored from
    row_inds_temp_begin = row_inds_temp_begin + (itPair.first - index_temp_begin);
    col_inds_temp_begin = col_inds_temp_begin + (itPair.first - index_temp_begin);
    index_temp_begin = itPair.first; // new_keys_end from unique_by_key
    values_temp_begin = itPair.second; // new_values_end from unique_by_key
    
    // update iterator where the "latest" row starts since smaller rows are not touched from now on
    auto row_find = thrust::find(typename ThrustSystem<A>::execution_policy(),
                                 row_inds_temp_begin - (index_temp_begin - index_begin), // search_begin
                                 row_inds_temp_begin, // search_end
                                 *(row_inds_temp_begin - 1)); // value to search for
    
    index_begin = index_temp_begin - (row_inds_temp_begin - row_find);
    values_begin = values_temp_begin - (row_inds_temp_begin - row_find);
    
    // update iterator for not yet used points
    point_temp_begin = point_temp_begin + nPointsToHandle;
  }
  
  NrPoints matrixSize = values_temp_begin - valuesVector.begin();
  
  // resize res
  res.rowIndices_.resize(matrixSize);
  res.columnIndices_.resize(matrixSize);
  valuesVector.resize(matrixSize);
  res.rowIndices_.shrink_to_fit();
  res.columnIndices_.shrink_to_fit();
  valuesVector.shrink_to_fit();
  
  NrBoxes nBoxes = this->count(depth);
  
  res.nRows_ = nBoxes;
  res.nColumns_ = nBoxes;
  
  // perhaps order row-major or perhaps omit getInvalidNrBoxes() as row
  if (useRowMajor) {
    bool areThereInvalidNrBoxes = thrust::any_of(typename ThrustSystem<A>::execution_policy(),
                                                 res.columnIndices_.begin(), res.columnIndices_.end(),
                                                 thrust::placeholders::_1 == getInvalidNrBoxes());
    
    if (areThereInvalidNrBoxes) {
      if (useInvalidNrBoxes) {
        res.nColumns_ += NrBoxes(1);
        
        thrust::replace_if(typename ThrustSystem<A>::execution_policy(),
                           res.columnIndices_.begin(), res.columnIndices_.end(), // input
                           thrust::placeholders::_1 == getInvalidNrBoxes(), // predicate
                           res.nColumns_);
      } else {
        typename ThrustSystem<A>::Vector<bool> stencil(res.columnIndices_.size());
        
        thrust::transform(typename ThrustSystem<A>::execution_policy(),
                          res.columnIndices_.begin(), res.columnIndices_.end(), // input
                          stencil.begin(), // output
                          thrust::placeholders::_1 == getInvalidNrBoxes());
                          
        
        auto zi_begin = thrust::make_zip_iterator(thrust::make_tuple(res.rowIndices_.begin(),
                                                                     res.columnIndices_.begin(),
                                                                     valuesVector.begin()));
        
        auto zi_end = thrust::remove_if(typename ThrustSystem<A>::execution_policy(),
                                        zi_begin, zi_begin + res.rowIndices_.size(), // entries to perhaps remove
                                        stencil.begin(), // stencil
                                        thrust::identity<bool>()); // predicate for stencil
        
        NrBoxes newSize = zi_end - zi_begin;
        res.rowIndices_.resize(newSize);
        res.columnIndices_.resize(newSize);
        valuesVector.resize(newSize);
        res.rowIndices_.shrink_to_fit();
        res.columnIndices_.shrink_to_fit();
        valuesVector.shrink_to_fit();
      }
    }
  } else {
    sortColumnMajorByIterators<A>(res.rowIndices_.begin(), res.columnIndices_.begin(),
                                  valuesVector.begin(),
                                  valuesVector.size());
    
    // after sorting getInvalidNrBoxes() can only stand at the end
    auto col_find = thrust::find(typename ThrustSystem<A>::execution_policy(),
                                 res.columnIndices_.begin(), // search_begin
                                 res.columnIndices_.end(), // search_end
                                 getInvalidNrBoxes()); // value to search for
    
    if (col_find != res.columnIndices_.end()) {
      if (useInvalidNrBoxes) {
        res.nColumns_ += NrBoxes(1);
        
        thrust::fill(typename ThrustSystem<A>::execution_policy(),
                     col_find, res.columnIndices_.end(), // in-place trafo
                     res.nColumns_);
      } else {
        NrBoxes newSize = col_find - res.columnIndices_.begin();
        res.rowIndices_.resize(newSize);
        res.columnIndices_.resize(newSize);
        valuesVector.resize(newSize);
        res.rowIndices_.shrink_to_fit();
        res.columnIndices_.shrink_to_fit();
        valuesVector.shrink_to_fit();
      }
    }
  }
  
  res.values_.resize(valuesVector.size());
  
  if (useAbsoluteValues) {
    res.values_.assign(valuesVector.begin(), valuesVector.end());
  } else {
    thrust::copy(typename ThrustSystem<A>::execution_policy(),
                 valuesVector.begin(), valuesVector.end(),
                 res.values_.begin());
    
    thrust::transform(typename ThrustSystem<A>::execution_policy(),
                      res.values_.begin(), res.values_.end(),
                      res.values_.begin(), // in-place trafo
                      thrust::placeholders::_1 / T(nPointsPerBox));
  }
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename T, typename Grid, typename Map>
inline CooMatrix<A, T> ImplicitBoxTree<A, DIM, REAL, N>::transitionMatrixForRoot(Grid grid, Map map,
                                                                                 bool useAbsoluteValues,
                                                                                 bool useRowMajor,
                                                                                 bool useInvalidNrBoxes)
{
  CooMatrix<A, T> res;
  
  this->transitionMatrix(this->makePointIteratorPairOverRoot(grid, map),
                         0, grid.getNumberOfPointsPerBox(),
                         useAbsoluteValues, useRowMajor, useInvalidNrBoxes,
                         res);
  
  return res;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename T, typename Grid, typename Map>
inline CooMatrix<A, T> ImplicitBoxTree<A, DIM, REAL, N>::transitionMatrixForLeaves(Grid grid, Map map, 
                                                                                   bool useAbsoluteValues,
                                                                                   bool useRowMajor,
                                                                                   bool useInvalidNrBoxes)
{
  CooMatrix<A, T> res;
  
  this->transitionMatrix(this->makePointIteratorPairOverLeaves(grid, map),
                         getLeafDepth(), grid.getNumberOfPointsPerBox(),
                         useAbsoluteValues, useRowMajor, useInvalidNrBoxes,
                         res);
  
  return res;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename T, typename Grid, typename Map>
inline CooMatrix<A, T> ImplicitBoxTree<A, DIM, REAL, N>::transitionMatrixForDepth(Depth depth,
                                                                                  Grid grid, Map map,
                                                                                  bool useAbsoluteValues,
                                                                                  bool useRowMajor,
                                                                                  bool useInvalidNrBoxes)
{
  CooMatrix<A, T> res;
  
  this->transitionMatrix(this->makePointIteratorPairOverDepth(depth, grid, map),
                         depth, grid.getNumberOfPointsPerBox(),
                         useAbsoluteValues, useRowMajor, useInvalidNrBoxes,
                         res);
  
  return res;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline NrBoxes ImplicitBoxTree<A, DIM, REAL, N>::unsetFlags(Flags flags, Depth depth)
{
  return this->unsetFlagsByStencil(thrust::make_constant_iterator(true), flags, depth);
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename BoxIteratorPair>
inline NrBoxes ImplicitBoxTree<A, DIM, REAL, N>::unsetFlagsByIterators(BoxIteratorPair points,
                                                                       Flags flags, Depth depth)
{
  return this->unsetFlagsBySearch(this->searchCompacted(points, depth), flags, depth);
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline NrBoxes ImplicitBoxTree<A, DIM, REAL, N>::unsetFlagsByPoints(
    const typename ThrustSystem<A>::Vector<REAL>& points, Flags flags, Depth depth)
{
  auto ti_begin =
      thrust::make_transform_iterator(
          thrust::make_transform_iterator(
              thrust::make_transform_iterator(thrust::make_counting_iterator(NrPoints(0)),
                                              thrust::placeholders::_1 * NrPoints(DIM)),
              AdvanceConstRealPointerFunctor<REAL>(thrust::raw_pointer_cast(points.data()))),
          ComputeBoxFromPointFunctor<DIM, REAL, N>(thrust::raw_pointer_cast(center_.data()), 
                                                   thrust::raw_pointer_cast(radius_.data()), 
                                                   thrust::raw_pointer_cast(sdScheme_.data()),
                                                   thrust::raw_pointer_cast(sdCount_.data()),
                                                   this->isSdSchemeDefault(),
                                                   Flag::NONE));
  
  return this->unsetFlagsByIterators(thrust::make_pair(ti_begin, ti_begin + points.size() / DIM), flags, depth);
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline NrBoxes ImplicitBoxTree<A, DIM, REAL, N>::unsetFlagsBySearch(
    const typename ThrustSystem<A>::Vector<NrBoxes>& searchVector, Flags flags, Depth depth)
{
  // if last element is getInvalidNrBoxes(), then reduce later the end()-iterator
  NrBoxes isLastElementInvalid = searchVector.size() ? searchVector.back() == getInvalidNrBoxes() : 0;
    
  if (depth == 0) {
    
    bool isSurroundingBoxHit = thrust::any_of(typename ThrustSystem<A>::execution_policy(),
                                              searchVector.begin(), searchVector.end() - isLastElementInvalid,
                                              thrust::placeholders::_1 == NrBoxes(0));
    
    bool haveAnyBoxesFlagsSet = thrust::any_of(typename ThrustSystem<A>::execution_policy(),
                                               flagsVector_.begin(), flagsVector_.end(),
                                               IsAnyFlagSetFunctor(flags));
    
    if (isSurroundingBoxHit && haveAnyBoxesFlagsSet) {
      
      thrust::transform(typename ThrustSystem<A>::execution_policy(),
                        flagsVector_.begin(), flagsVector_.end(),
                        flagsVector_.begin(),
                        UnsetFlagsFunctor(flags));
      
      if (isAnyFlagSet(flagsToCheckForLeafIndices_, flags)) {
        this->freeLeafIndicesFromDepth();
      }
      
      return 1;
      
    } else {
      
      return 0;
      
    }
    
  } else if (depth == getLeafDepth() || this->areAllLeavesOnDepth(depth)) {
    
    NrBoxes res = thrust::count_if(typename ThrustSystem<A>::execution_policy(),
                                   thrust::make_permutation_iterator(flagsVector_.begin(),
                                                                     searchVector.begin()), // begin
                                   thrust::make_permutation_iterator(flagsVector_.begin(),
                                                                     searchVector.end() - isLastElementInvalid), // end
                                   IsAnyFlagSetFunctor(flags)); // count boxes that have at least one Flag of flags set
    
    thrust::transform_if(typename ThrustSystem<A>::execution_policy(),
                         thrust::make_permutation_iterator(flagsVector_.begin(), searchVector.begin()), // begin
                         thrust::make_permutation_iterator(flagsVector_.begin(),
                                                           searchVector.end() - isLastElementInvalid), // end
                         searchVector.begin(), // stencil
                         thrust::make_permutation_iterator(flagsVector_.begin(),
                                                           searchVector.begin()), // in-place trafo
                         UnsetFlagsFunctor(flags), // operation to execute
                         thrust::placeholders::_1 != getInvalidNrBoxes()); // predicate for stencil
    
    if (res > 0 && isAnyFlagSet(flagsToCheckForLeafIndices_, flags)) {
      this->freeLeafIndicesFromDepth();
    }
    
    return res;
    
  } else if (depth <= this->depth()) {
    
    // fill stencilVector with "false"
    typename ThrustSystem<A>::Vector<bool> stencilVector(this->count(depth), false);
    
    thrust::transform_if(typename ThrustSystem<A>::execution_policy(),
                         thrust::make_permutation_iterator(stencilVector.begin(), searchVector.begin()), // begin
                         thrust::make_permutation_iterator(stencilVector.begin(),
                                                           searchVector.end() - isLastElementInvalid), // end
                         searchVector.begin(), // stencil for trafo
                         thrust::make_permutation_iterator(stencilVector.begin(),
                                                           searchVector.begin()), // in-place trafo
                         thrust::logical_not<bool>(), // stencilVector (false) -> not(stencilVector) (true)
                         thrust::placeholders::_1 != getInvalidNrBoxes()); // predicate for stencil
    
    NrBoxes res = this->unsetFlagsByStencil(stencilVector.begin(), flags, depth);
    
    return res;
    
  } else {
    
    return 0;
    
  }
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename BoolIterator>
inline NrBoxes ImplicitBoxTree<A, DIM, REAL, N>::unsetFlagsByStencil(BoolIterator stencil_begin,
                                                                     Flags flags, Depth depth)
{
  NrBoxes nBoxes = this->countIfFlags(IsAnyFlagSetFunctor(flags), FlagsOrFunctor(), stencil_begin, depth);
  
  if (depth == 0) {
    
    if (*stencil_begin) { // must stand inside then-branch!
      thrust::transform(typename ThrustSystem<A>::execution_policy(),
                        flagsVector_.begin(), flagsVector_.end(),
                        flagsVector_.begin(),
                        UnsetFlagsFunctor(flags));
    }
    
  } else if (depth == getLeafDepth() || this->areAllLeavesOnDepth(depth)) {
    
    thrust::transform_if(typename ThrustSystem<A>::execution_policy(),
                         flagsVector_.begin(), flagsVector_.end(),
                         stencil_begin, // stencil
                         flagsVector_.begin(), // in-place trafo
                         UnsetFlagsFunctor(flags),
                         thrust::identity<bool>()); // predicate for stencil
    
  } else if (depth == getUnsubdividableDepth()) {
    
    // initialise boxes' indeces in decremented Depth if needed
    this->initializeIndicesInDepth(depth);
    
    // pi_begin[i] == stencil_begin[indicesInDepth_[i]]
    auto pi_begin = thrust::make_permutation_iterator(stencil_begin, indicesInDepth_.begin());
    
    // store values since stencil and in-/output range would otherwise overlap in later transform_if
    typename ThrustSystem<A>::Vector<bool> isUnsubdividableVector(this->isUnsubdividable_begin(),
                                                                  this->isUnsubdividable_begin() + this->size());
    
    // fusing stencil and isUnsubdividable
    auto zi_begin = thrust::make_zip_iterator(thrust::make_tuple(pi_begin,
                                                                 isUnsubdividableVector.begin()));
    
    // unset Flags flags in boxes b if b is unsubdividable and stencil_begin[indicesInDepth_[*b]] is true
    thrust::transform_if(typename ThrustSystem<A>::execution_policy(),
                         flagsVector_.begin(), flagsVector_.end(),
                         zi_begin, // stencil
                         flagsVector_.begin(), // in-place trafo
                         UnsetFlagsFunctor(flags),
                         AllIn2TupleFunctor()); // logical and in tuple of stencil; predicate for stencil
    
  } else if (depth <= this->depth()) {
    
    // initialise boxes' indeces in decremented Depth if needed
    this->initializeIndicesInDepth(depth);
    
    // pi_begin[i] == stencil_begin[indicesInDepth_[i]]
    auto pi_begin = thrust::make_permutation_iterator(stencil_begin, indicesInDepth_.begin());
    
    // depth_begin[i] == hasAtLeastDepth(this[i])
    auto depth_begin = thrust::make_transform_iterator(this->begin(), HasAtLeastDepthFunctor<N>(depth));
    
    // fusing stencil, hasAtLeastDepth and haveAllBoxesInSegmentFlagsSet
    auto zi_begin = thrust::make_zip_iterator(thrust::make_tuple(pi_begin, depth_begin));
    
    // unset Flags flags in boxes b if b has at least Depth depth and stencil_begin[indicesInDepth_[*b]] is true
    thrust::transform_if(typename ThrustSystem<A>::execution_policy(),
                         flagsVector_.begin(), flagsVector_.end(),
                         zi_begin, // stencil
                         flagsVector_.begin(), // in-place trafo
                         UnsetFlagsFunctor(flags),
                         AllIn2TupleFunctor()); // logical and in tuple of stencil; predicate for stencil
  }
  
  if (nBoxes > 0 && isAnyFlagSet(flagsToCheckForLeafIndices_, flags)) {
    this->freeLeafIndicesFromDepth();
  }
  
  return nBoxes;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline NrBoxes ImplicitBoxTree<A, DIM, REAL, N>::unsubdivide(Flags flags)
{
  // store values since stencil and in-/output range would otherwise overlap in later transform_if
  typename ThrustSystem<A>::Vector<bool> isUnsubdividableVector(this->isUnsubdividable_begin(flags),
                                                                this->isUnsubdividable_begin(flags) + this->size());
  
  // res = number of leaves to unsubdivide
  NrBoxes res = thrust::count_if(typename ThrustSystem<A>::execution_policy(),
                                 isUnsubdividableVector.begin(), isUnsubdividableVector.end(),
                                 thrust::identity<bool>());
  
  // unsubdivide if possible
  thrust::transform_if(typename ThrustSystem<A>::execution_policy(),
                       this->begin(), this->end(),
                       isUnsubdividableVector.begin(), // stencil
                       this->begin(), // in-place trafo
                       DecrementDepthByFunctor<N>(1),
                       thrust::identity<bool>()); // stencil -> stencil/bool
  
  isUnsubdividableVector.clear();
  isUnsubdividableVector.shrink_to_fit();
  
  // combine (and) flags of "equal" boxes
  thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(),
                                this->rbegin(), this->rend(), // keys
                                flagsVector_.rbegin(), // Note all reverse_iterators!
                                flagsVector_.rbegin(), // in-place trafo
                                IsSameBoxFunctor<N>(), // "segment operator" for keys
                                FlagsAndFunctor()); // first box of segment gets intersection of all flags
  
  auto box_old_end = this->end();
  
  // delete all of equal consecutive boxes but the first one
  // substract those boxes that had a sibling to prevent double counting
  auto box_new_end = thrust::unique(typename ThrustSystem<A>::execution_policy(),
                                    this->begin(), this->end(),
                                    IsSameBoxFunctor<N>());
  res -= box_old_end - box_new_end;
  
  this->resize(box_new_end - this->begin());
  
  if (res > 0) {
    this->freeIndicesInDepth();
    this->freeLeafIndicesFromDepth();
  }
  
  return res;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline const typename ThrustSystem<A>::Vector<REAL>& ImplicitBoxTree<A, DIM, REAL, N>::getCenter() const
{
  return center_;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename REAL2>
inline typename ThrustSystem<A>::Vector<REAL>& ImplicitBoxTree<A, DIM, REAL, N>::setCenter(
    const typename ThrustSystem<A>::Vector<REAL2>& center)
{
  center_.assign(center.begin(), center.end());
  // cut size to DIM, or extend to DIM by filling new entries with 0.5
  center_.resize(DIM, 0.5);
  return center_;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline const typename ThrustSystem<A>::Vector<REAL>& ImplicitBoxTree<A, DIM, REAL, N>::getRadius() const
{
  return radius_;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename REAL2>
inline typename ThrustSystem<A>::Vector<REAL>& ImplicitBoxTree<A, DIM, REAL, N>::setRadius(
    const typename ThrustSystem<A>::Vector<REAL2>& radius)
{
  // check whether all entries are non-negative
  bool isValidRadius = thrust::all_of(typename ThrustSystem<A>::execution_policy(),
                                      radius.begin(), radius.end(),
                                      thrust::placeholders::_1 >= REAL2(0.0));
  
  if (isValidRadius) {
    radius_.assign(radius.begin(), radius.end());
    // cut size to DIM, or extend to DIM by filling new entries with 0.5
    radius_.resize(DIM, 0.5);
  } else {
    std::cout << "Warning: Input has negative values. Radius vector remains unchanged." << std::endl;
  }
  return radius_;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline const typename ThrustSystem<A>::Vector<Dimension>& ImplicitBoxTree<A, DIM, REAL, N>::getSdScheme() const
{
  return sdScheme_;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
template<typename UInt>
inline typename ThrustSystem<A>::Vector<Dimension>& ImplicitBoxTree<A, DIM, REAL, N>::setSdScheme(
    const typename ThrustSystem<A>::Vector<UInt>& sdScheme)
{
  bool isValidSdScheme = thrust::all_of(typename ThrustSystem<A>::execution_policy(),
                                        sdScheme.begin(), sdScheme.end(),
                                        thrust::placeholders::_1 >= 0 && thrust::placeholders::_1 < DIM);
  
  if (isValidSdScheme) {
    // newly initialise
    sdScheme_ = typename ThrustSystem<A>::Vector<Dimension>(int(getMaxDepth<N>()) + 1);
    
    uint64_t nEntries = 0;
    
    if (sdScheme.size() > 0) {
      // if sdScheme.size() >= sdScheme_.size(), the first sdScheme_.size() entries of sdScheme are copied to sdScheme_,
      // else sdScheme is periodically stored in sdScheme_
      for (auto sd_it = sdScheme_.begin(); sd_it != sdScheme_.end(); sd_it += nEntries) {
        
        nEntries = thrust::minimum<uint64_t>()(sdScheme.size(), sdScheme_.end() - sd_it);
        
        thrust::copy(typename ThrustSystem<A>::execution_policy(),
                     sdScheme.begin(), sdScheme.begin() + nEntries,
                     sd_it);
      }
    } else {
      // sdScheme_ = (0, 1, …, DIM-1, 0, 1, …, DIM-1, …)
      sdScheme_.assign(thrust::make_transform_iterator(thrust::make_counting_iterator(Depth(0)),
                                                       thrust::placeholders::_1 % DIM),
                       thrust::make_transform_iterator(thrust::make_counting_iterator(getMaxDepth<N>()),
                                                       thrust::placeholders::_1 % DIM));
    }
    
    this->adaptSdCount();
  } else {
    std::cout << "Warning: Input has values >= DIM. SdScheme remains unchanged." << std::endl;
  }
    
  return sdScheme_;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline const typename ThrustSystem<A>::Vector<Depth>& ImplicitBoxTree<A, DIM, REAL, N>::getSdCount() const
{
  return sdCount_;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline const typename ThrustSystem<A>::Vector<Depth>& ImplicitBoxTree<A, DIM, REAL, N>::getDepthVector() const
{
  return depthVector_;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline const typename ThrustSystem<A>::Vector<Flags>& ImplicitBoxTree<A, DIM, REAL, N>::getFlagsVector() const
{
  return flagsVector_;
}
  
template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline uint64_t ImplicitBoxTree<A, DIM, REAL, N>::getNHostBufferBytes() const
{
  return nHostBufferBytes_;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline uint64_t ImplicitBoxTree<A, DIM, REAL, N>::setNHostBufferBytes(uint64_t nHostBufferBytes)
{
  nHostBufferBytes_ = nHostBufferBytes;
  return nHostBufferBytes_;
}


template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline typename ThrustSystem<A>::ConstBoxIterator<N> ImplicitBoxTree<A, DIM, REAL, N>::begin() const
{
  return thrust::make_zip_iterator(thrust::make_tuple(bitPatternVector_.begin(),
                                                      depthVector_.begin(),
                                                      flagsVector_.begin()));
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline typename ThrustSystem<A>::BoxIterator<N> ImplicitBoxTree<A, DIM, REAL, N>::begin()
{
  return thrust::make_zip_iterator(thrust::make_tuple(bitPatternVector_.begin(),
                                                      depthVector_.begin(),
                                                      flagsVector_.begin()));
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline typename ThrustSystem<A>::ConstBoxIterator<N> ImplicitBoxTree<A, DIM, REAL, N>::end() const
{
  return thrust::make_zip_iterator(thrust::make_tuple(bitPatternVector_.end(),
                                                      depthVector_.end(),
                                                      flagsVector_.end()));
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline typename ThrustSystem<A>::BoxIterator<N> ImplicitBoxTree<A, DIM, REAL, N>::end()
{
  return thrust::make_zip_iterator(thrust::make_tuple(bitPatternVector_.end(),
                                                      depthVector_.end(),
                                                      flagsVector_.end()));
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline thrust::reverse_iterator<typename ThrustSystem<A>::ConstBoxIterator<N>>
ImplicitBoxTree<A, DIM, REAL, N>::rbegin() const
{
  return thrust::make_reverse_iterator(this->end());
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline thrust::reverse_iterator<typename ThrustSystem<A>::BoxIterator<N>> ImplicitBoxTree<A, DIM, REAL, N>::rbegin()
{
  return thrust::make_reverse_iterator(this->end());
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline thrust::reverse_iterator<typename ThrustSystem<A>::ConstBoxIterator<N>>
ImplicitBoxTree<A, DIM, REAL, N>::rend() const
{
  return thrust::make_reverse_iterator(this->begin());
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline thrust::reverse_iterator<typename ThrustSystem<A>::BoxIterator<N>> ImplicitBoxTree<A, DIM, REAL, N>::rend()
{
  return thrust::make_reverse_iterator(this->begin());
}

template<Dimension DIM, typename REAL, Depth N>
inline std::ostream& operator<<(std::ostream& os, const ImplicitBoxTree<HOST, DIM, REAL, N>& implicitBoxTree)
{
  REAL vec[2 * DIM + 1];
  
  ComputeCoordinatesOfBoxFunctor<DIM, REAL, N> fun(thrust::raw_pointer_cast(implicitBoxTree.getCenter().data()),
                                                   thrust::raw_pointer_cast(implicitBoxTree.getRadius().data()),
                                                   thrust::raw_pointer_cast(implicitBoxTree.getSdScheme().data()),
                                                   thrust::raw_pointer_cast(implicitBoxTree.getSdCount().data()),
                                                   implicitBoxTree.isSdSchemeDefault());
  
  for (auto _begin = implicitBoxTree.begin(); _begin < implicitBoxTree.end(); ++_begin) {
    
    os << getBitPattern<N>(*_begin) << "\t" << getDepth<N>(*_begin) << "\t" << getFlags<N>(*_begin) << "\t";
    
    fun(thrust::make_tuple(*_begin, (REAL *) vec));
    
    for (int i = 0; i < 2*DIM + 1; ++i) {
      os << vec[i] << "\t";
    }
    
    os << "\n";
  }
  return os;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline std::ostream& operator<<(std::ostream& os, const ImplicitBoxTree<A, DIM, REAL, N>& implicitBoxTree)
{
  os << ImplicitBoxTree<HOST, DIM, REAL, N>(implicitBoxTree);
  return os;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline Dimension ImplicitBoxTree<A, DIM, REAL, N>::dim() const
{
  return DIM;
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline void ImplicitBoxTree<A, DIM, REAL, N>::reserve(NrBoxes n)
{
  if (n != getInvalidNrBoxes()) {  
    // "If n is less than or equal to capacity(), this call has no effect.
    // Otherwise, this method is a request for allocation of additional memory.
    // If the request is successful, then capacity() is greater than or equal to n;
    // otherwise, capacity() is unchanged. In either case, size() is unchanged."
    bitPatternVector_.reserve(n);
    depthVector_.reserve(n);
    flagsVector_.reserve(n);
  } else {
    std::cout << "Warning: Invalid input. Nothing is changed." << std::endl;
  }
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline void ImplicitBoxTree<A, DIM, REAL, N>::resize(NrBoxes n)
{
  if (n != getInvalidNrBoxes()) {  
    // "This method will resize this vector to the specified number of elements.
    // If the number is smaller than this vector's current size this vector is truncated,
    // otherwise this vector is extended and new elements are populated with given data."
    // capacities are (actually) not reduced
    if (n != this->size()) {
      this->freeIndicesInDepth();
      this->freeLeafIndicesFromDepth();
    }
    bitPatternVector_.resize(n);
    depthVector_.resize(n);
    flagsVector_.resize(n);
  } else {
    std::cout << "Warning: Invalid input. Nothing is changed." << std::endl;
  }
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline void ImplicitBoxTree<A, DIM, REAL, N>::shrinkToFit()
{
  // "This method shrinks the capacity of this vector to exactly fit its elements."
  bitPatternVector_.shrink_to_fit();
  depthVector_.shrink_to_fit();
  flagsVector_.shrink_to_fit();
}

template<Architecture A, Dimension DIM, typename REAL, Depth N>
inline NrBoxes ImplicitBoxTree<A, DIM, REAL, N>::size() const
{
  return bitPatternVector_.size();
}

} // namespace b12
