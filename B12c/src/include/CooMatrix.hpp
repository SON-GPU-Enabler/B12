
#include <fstream>

#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/unique.h>

#include "Functors.h"
#include "helpFunctions.h"


namespace b12 {

template<Architecture A>
inline void compressIndexVector(typename ThrustSystem<A>::Vector<NrBoxes>& indices, NrBoxes dim)
{
  if (uint64_t(indices.size()) > uint64_t(NrBoxes(-1))) {
    
    std::cout << "Warning: The length of the given vector exceeds the underlying type's range. Nothing is changed."
              << std::endl;
              
  } else {
    
    typename ThrustSystem<A>::Vector<NrBoxes> entryCount(indices.size(), 1);
    
    // first entry in segment gets count of the corresponding index
    thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(),
                                  indices.rbegin(), indices.rend(), // keys_input
                                  entryCount.rbegin(), // Note all reverse_iterators!
                                  entryCount.rbegin()); // in-place trafo
    
    // only retain first entry in each segment
    auto itPair = thrust::unique_by_key(typename ThrustSystem<A>::execution_policy(),
                                        indices.begin(), indices.end(), // keys
                                        entryCount.begin()); // values
    
    // resize for more memory; but not changing indices' size might be more efficient
    entryCount.resize(itPair.second - entryCount.begin());
    
    typename ThrustSystem<A>::Vector<NrBoxes> temp(dim + 1, 0);
    
    // temp[indices[i] + 1] = entryCount[i]
    thrust::scatter(typename ThrustSystem<A>::execution_policy(),
                    entryCount.begin(), entryCount.end(), // values to copy
                    indices.begin(), // map for storing in output
                    temp.begin() + 1); // output; omit first value since it must be 0 by design
    
    entryCount.clear();
    indices.resize(dim + 1);
    
    thrust::inclusive_scan(typename ThrustSystem<A>::execution_policy(),
                           temp.begin(), temp.end(), // values to sum up
                           indices.begin()); // output
    
  }
}

template<Architecture A>
inline void extendCompressedIndexVector(typename ThrustSystem<A>::Vector<NrBoxes>& indices)
{
  // vector with length indices.size(), filled with 1's
  typename ThrustSystem<A>::Vector<NrBoxes> temp(indices.size(), 1);
  
  // result vector of size nnz filled with 0's
  typename ThrustSystem<A>::Vector<NrBoxes> res(indices.back(), 0);
  
  // get the step for consecutive indices (e.g. if one row/column is empty, the index jumps by 2 instead of 1)
  thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(),
                                indices.rbegin(), indices.rend(), // keys_input
                                temp.rbegin(), // Note all reverse_iterators!
                                temp.rbegin()); // in-place trafo
  
  // only retain first entry in each segment; last element is not needed since it is no actual row/column
  auto itPair = thrust::unique_by_key(typename ThrustSystem<A>::execution_policy(),
                                      indices.begin(), indices.end(), // keys
                                      temp.begin()); // values
  
  // res[indices[i]] = temp[i]; last value is not used since it is not an actual row/column
  thrust::scatter(typename ThrustSystem<A>::execution_policy(),
                  temp.begin(), temp.end() - 1, // values to copy
                  indices.begin(), // map for storing in output
                  res.begin()); // output; omit first value since it must be 0 by design
  
  temp.clear();
  indices.resize(res.size());
  
  // decrement result vector's first element; otherwise it would not correspond to 0-based indexing
  --res[0];
  
  thrust::inclusive_scan(typename ThrustSystem<A>::execution_policy(),
                         res.begin(), res.end(), // values to sum up
                         indices.begin()); // output
}

template<Architecture A, typename IndexIterator, typename ValuesIterator>
inline void sortColumnMajorByIterators(IndexIterator row_begin, IndexIterator column_begin,
                                       ValuesIterator values_begin, NrPoints n, std::true_type)
{
    auto index_begin = thrust::make_zip_iterator(thrust::make_tuple(column_begin, row_begin));
    
    typename ThrustSystem<A>::Vector<NrPoints> keys(n);
    
    thrust::transform(typename ThrustSystem<A>::execution_policy(),
                      index_begin, index_begin + n,
                      keys.begin(),
                      PackIntegersFunctor<NrBoxes, NrPoints>()); // column precedes row
    
    thrust::sort_by_key(typename ThrustSystem<A>::execution_policy(),
                        keys.begin(), keys.end(),
                        values_begin);
    
    thrust::transform(typename ThrustSystem<A>::execution_policy(),
                      keys.begin(), keys.end(),
                      index_begin,
                      UnpackIntegersFunctor<NrBoxes, NrPoints>());
}

template<Architecture A, typename IndexIterator, typename ValuesIterator>
inline void sortColumnMajorByIterators(IndexIterator row_begin, IndexIterator column_begin,
                                       ValuesIterator values_begin, NrPoints n, std::false_type)
{
  auto index_begin = thrust::make_zip_iterator(thrust::make_tuple(row_begin, column_begin));
    
  thrust::sort_by_key(typename ThrustSystem<A>::execution_policy(),
                      index_begin, index_begin + n, // keys
                      values_begin, // values to sort
                      ColumnMajorOrderingFunctor());
}

template<Architecture A, typename IndexIterator, typename ValuesIterator>
inline void sortColumnMajorByIterators(IndexIterator row_begin, IndexIterator column_begin,
                                       ValuesIterator values_begin, NrPoints n)
{
  // if packing is possible, use radix sort, else merge sort
  sortColumnMajorByIterators<A>(row_begin, column_begin, values_begin, n,
                                PackIntegersFunctor<NrBoxes, NrPoints>::IsValid());
}

template<Architecture A, typename IndexIterator, typename ValuesIterator>
inline void sortRowMajorByIterators(IndexIterator row_begin, IndexIterator column_begin,
                                    ValuesIterator values_begin, NrPoints n, std::true_type)
{
    auto index_begin = thrust::make_zip_iterator(thrust::make_tuple(row_begin, column_begin));
    
    typename ThrustSystem<A>::Vector<NrPoints> keys(n);
    
    thrust::transform(typename ThrustSystem<A>::execution_policy(),
                      index_begin, index_begin + n,
                      keys.begin(),
                      PackIntegersFunctor<NrBoxes, NrPoints>()); // row precedes column
    
    thrust::sort_by_key(typename ThrustSystem<A>::execution_policy(),
                        keys.begin(), keys.end(),
                        values_begin);
    
    thrust::transform(typename ThrustSystem<A>::execution_policy(),
                      keys.begin(), keys.end(),
                      index_begin,
                      UnpackIntegersFunctor<NrBoxes, NrPoints>());
}

template<Architecture A, typename IndexIterator, typename ValuesIterator>
inline void sortRowMajorByIterators(IndexIterator row_begin, IndexIterator column_begin,
                                    ValuesIterator values_begin, NrPoints n, std::false_type)
{
  auto index_begin = thrust::make_zip_iterator(thrust::make_tuple(row_begin, column_begin));
    
  thrust::sort_by_key(typename ThrustSystem<A>::execution_policy(),
                      index_begin, index_begin + n, // keys
                      values_begin, // values to sort
                      RowMajorOrderingFunctor());
}

template<Architecture A, typename IndexIterator, typename ValuesIterator>
inline void sortRowMajorByIterators(IndexIterator row_begin, IndexIterator column_begin,
                                    ValuesIterator values_begin, NrPoints n)
{
  // if packing is possible, use radix sort, else merge sort
  sortRowMajorByIterators<A>(row_begin, column_begin, values_begin, n,
                             PackIntegersFunctor<NrBoxes, NrPoints>::IsValid());  
}

__host__ __device__
inline bool ColumnMajorOrderingFunctor::operator()(MatrixIndex mi1, MatrixIndex mi2) const
{
  return thrust::get<1>(mi1) < thrust::get<1>(mi2)
         ||
         thrust::get<1>(mi1) == thrust::get<1>(mi2) && thrust::get<0>(mi1) < thrust::get<0>(mi2);
}

__host__ __device__
inline bool RowMajorOrderingFunctor::operator()(MatrixIndex mi1, MatrixIndex mi2) const
{
  return thrust::get<0>(mi1) < thrust::get<0>(mi2)
         ||
         thrust::get<0>(mi1) == thrust::get<0>(mi2) && thrust::get<1>(mi1) < thrust::get<1>(mi2);
}

template<Architecture A, typename T>
inline CooMatrix<A, T>::CooMatrix(NrBoxes nRows, NrBoxes nColumns, uint64_t capacity)
    : rowIndices_(capacity), columnIndices_(capacity), values_(capacity),
      nRows_(nRows), nColumns_(nColumns) {}

template<Architecture A, typename T>
template<Architecture A2, typename T2>
inline CooMatrix<A, T>::CooMatrix(const CooMatrix<A2, T2>& cooMatrix) 
    : rowIndices_(cooMatrix.rowIndices_.size()), columnIndices_(cooMatrix.columnIndices_.size()),
      values_(cooMatrix.values_.size()), nRows_(cooMatrix.nRows_), nColumns_(cooMatrix.nColumns_)
{
  rowIndices_.assign(cooMatrix.rowIndices_.begin(), cooMatrix.rowIndices_.end());
  columnIndices_.assign(cooMatrix.columnIndices_.begin(), cooMatrix.columnIndices_.end());
  values_.assign(cooMatrix.values_.begin(), cooMatrix.values_.end());
}

template<Architecture A, typename T>
template<typename T2>
inline CooMatrix<A, T>::CooMatrix(const typename ThrustSystem<A>::Vector<NrBoxes>& rowIndices,
                                  const typename ThrustSystem<A>::Vector<NrBoxes>& columnIndices,
                                  const typename ThrustSystem<A>::Vector<T2>& values,
                                  NrBoxes nRows, NrBoxes nColumns, uint64_t capacity)
    : CooMatrix(rowIndices.begin(), columnIndices.begin(), values.begin(),
                rowIndices.size(), nRows, nColumns, capacity) {}

template<Architecture A, typename T>
template<typename NrBoxesIterator, typename ValuesIterator>
inline CooMatrix<A, T>::CooMatrix(NrBoxesIterator row_begin, NrBoxesIterator column_begin,
                                  ValuesIterator values_begin, uint64_t nValues,
                                  NrBoxes nRows, NrBoxes nColumns, uint64_t capacity)
    : rowIndices_(row_begin, row_begin + nValues), columnIndices_(column_begin, column_begin + nValues),
      values_(values_begin, values_begin + nValues), nRows_(nRows), nColumns_(nColumns)
{
  this->reserve(capacity);
}

template<Architecture A, typename T>
inline CooMatrix<A, T>::CooMatrix(const std::string& fileName)
    : CooMatrix()
{
  bool isOpenable, wasSuccess;
  
  std::ifstream ifs;
  
  ifs.open(fileName, std::ifstream::in);
  if (ifs.good()) {
    isOpenable = true;
  } else {
    // try to open fileName+".json"
    ifs.close();
    ifs.open(fileName + ".json", std::ifstream::in);
    isOpenable = ifs.good();
  }
  
  if (isOpenable) {
    std::string fileString;
    if (ifs.good()) {
      std::stringstream strStream;
      strStream << ifs.rdbuf();
      fileString = strStream.str();
    }
    wasSuccess = ifs.good();
    ifs.close();
    
    nRows_ = getNumberFromJSONString<NrBoxes>(fileString, "nRows");
    
    nColumns_ = getNumberFromJSONString<NrBoxes>(fileString, "nColumns");
    
    uint64_t nnz = getNumberFromJSONString<NrBoxes>(fileString, "nnz");
    
    {
      auto _rowIndices = getNumberVectorFromJSONString<NrBoxes>(fileString, "rowIndices", nnz);
      rowIndices_.assign(_rowIndices.begin(), _rowIndices.end());
    }
    
    {
      auto _columnIndices = getNumberVectorFromJSONString<NrBoxes>(fileString, "columnIndices", nnz);
      columnIndices_.assign(_columnIndices.begin(), _columnIndices.end());
    }
    
    {
      auto _values = getNumberVectorFromJSONString<T>(fileString, "values", nnz);
      values_.assign(_values.begin(), _values.end());
    }
    
    if (! wasSuccess) {
      std::cout << "Warning: An error occurred during reading." << std::endl;
    }
  } else {
    std::cout << "Warning: File could not be opened." << std::endl;
  }
}

template<Architecture A, typename T>
template<Architecture A2, typename T2>
inline CooMatrix<A, T>& CooMatrix<A, T>::operator=(const CooMatrix<A2, T2>& cooMatrix)
{
  rowIndices_.assign(cooMatrix.rowIndices_.begin(), cooMatrix.rowIndices_.end());
  columnIndices_.assign(cooMatrix.columnIndices_.begin(), cooMatrix.columnIndices_.end());
  values_.assign(cooMatrix.values_.begin(), cooMatrix.values_.end());
  
  nRows_ = cooMatrix.nRows_;
  nColumns_ = cooMatrix.nColumns_;
}

template<Architecture A, typename T>
template<Architecture A2, typename T2>
inline bool CooMatrix<A, T>::operator==(const CooMatrix<A2, T2>& cooMatrix) const
{
  bool res = nRows_ == cooMatrix.nRows_ && nColumns_ == cooMatrix.nColumns_;
  
  if (res) {
    if (A == A2) {
      bool isFirstSortedRowMajor = this->isSortedRowMajor();
      bool isFirstSortedColumnMajor = isFirstSortedRowMajor ? false : this->isSortedColumnMajor();
      bool isSecondSortedRowMajor = cooMatrix.isSortedRowMajor();
      bool isSecondSortedColumnMajor = isSecondSortedRowMajor ? false : cooMatrix.isSortedColumnMajor();
      
      // if one matrix is ordered and the other one is not, compare with a temporary properly ordered matrix
      if (isFirstSortedRowMajor && !isSecondSortedRowMajor) {
        res = *this == CooMatrix<A, T2>(cooMatrix).sortRowMajor();
      } else if (isFirstSortedColumnMajor && !isSecondSortedColumnMajor) {
        res = *this == CooMatrix<A, T2>(cooMatrix).sortColumnMajor();
      } else if (isSecondSortedRowMajor && !isFirstSortedRowMajor) {
        res = CooMatrix<A, T>(*this).sortRowMajor() == cooMatrix;
      } else if (isSecondSortedColumnMajor && !isFirstSortedColumnMajor) {
        res = CooMatrix<A, T>(*this).sortColumnMajor() == cooMatrix;
      } else {
        // both matrices are either ordered in the same way or are not ordered at all
        res = thrust::equal(typename ThrustSystem<A>::execution_policy(),
                            rowIndices_.begin(), rowIndices_.end(),
                            cooMatrix.rowIndices_.begin()) &&
              thrust::equal(typename ThrustSystem<A>::execution_policy(),
                            columnIndices_.begin(), columnIndices_.end(),
                            cooMatrix.columnIndices_.begin()) &&
              thrust::equal(typename ThrustSystem<A>::execution_policy(),
                            values_.begin(), values_.end(),
                            cooMatrix.values_.begin());
        
        // if res == true, fine,
        // else: if both matrices are not ordered in the same way, compare sorted copies
        if (!res && 
            !(isFirstSortedRowMajor && isSecondSortedRowMajor || 
              isFirstSortedColumnMajor && isSecondSortedColumnMajor)) {
          res = CooMatrix<A, T>(*this).sortRowMajor() == CooMatrix<A, T2>(cooMatrix).sortRowMajor();
        }
      }
    } else {
      res = *this == CooMatrix<A, T2>(cooMatrix);
    }
  }
  
  return res;
}

template<Architecture A, typename T>
inline bool CooMatrix<A, T>::isSortedColumnMajor() const
{
  return thrust::is_sorted(typename ThrustSystem<A>::execution_policy(),
                           thrust::make_zip_iterator(thrust::make_tuple(rowIndices_.begin(),
                                                                        columnIndices_.begin())), // keys_begin
                           thrust::make_zip_iterator(thrust::make_tuple(rowIndices_.end(),
                                                                        columnIndices_.end())), // keys_end
                           ColumnMajorOrderingFunctor());
}

template<Architecture A, typename T>
inline bool CooMatrix<A, T>::isSortedRowMajor() const
{
  return thrust::is_sorted(typename ThrustSystem<A>::execution_policy(),
                           thrust::make_zip_iterator(thrust::make_tuple(rowIndices_.begin(),
                                                                        columnIndices_.begin())), // keys_begin
                           thrust::make_zip_iterator(thrust::make_tuple(rowIndices_.end(),
                                                                        columnIndices_.end())), // keys_end
                           RowMajorOrderingFunctor());
}

template<Architecture A, typename T>
inline void CooMatrix<A, T>::makeSymmetric(bool makeAntisymmetric)
{
  uint64_t nnz = rowIndices_.size();
  bool wasSortedRowMajor = this->isSortedRowMajor();
  
  rowIndices_.resize(nnz * 2);
  columnIndices_.resize(nnz * 2);
  values_.resize(nnz * 2);
  
  thrust::copy(rowIndices_.begin(), rowIndices_.begin() + nnz, columnIndices_.begin() + nnz);
  thrust::copy(columnIndices_.begin(), columnIndices_.begin() + nnz, rowIndices_.begin() + nnz);
  
  if (makeAntisymmetric) {
    thrust::transform(typename ThrustSystem<A>::execution_policy(),
                      values_.begin(), values_.begin() + nnz,
                      values_.begin() + nnz,
                      -thrust::placeholders::_1);
  } else {
    thrust::copy(values_.begin(), values_.begin() + nnz,
                 values_.begin() + nnz);
  }
  
  if (wasSortedRowMajor) {
    this->sortRowMajor();
  } else {
    this->sortColumnMajor();
  }
  
  auto inds_begin = thrust::make_zip_iterator(thrust::make_tuple(rowIndices_.begin(), columnIndices_.begin()));
  
  thrust::inclusive_scan_by_key(typename ThrustSystem<A>::execution_policy(),
                                thrust::make_reverse_iterator(inds_begin + nnz * 2),
                                thrust::make_reverse_iterator(inds_begin),
                                values_.rbegin(),
                                values_.rbegin());
  
  auto itPair = thrust::unique_by_key(typename ThrustSystem<A>::execution_policy(),
                                      inds_begin, inds_begin + nnz * 2,
                                      values_.begin());
  
  uint64_t newSize = itPair.first - inds_begin;
  
  rowIndices_.resize(newSize);
  columnIndices_.resize(newSize);
  values_.resize(newSize);
  
  thrust::transform(typename ThrustSystem<A>::execution_policy(),
                    values_.begin(), values_.end(),
                    values_.begin(),
                    thrust::placeholders::_1 / T(2));
}

template<Architecture A, typename T>
inline CooMatrix<A, T>& CooMatrix<A, T>::sortColumnMajor()
{
  sortColumnMajorByIterators<A>(rowIndices_.begin(), columnIndices_.begin(),
                                values_.begin(),
                                values_.size());
  
  return *this;
}

template<Architecture A, typename T>
inline CooMatrix<A, T>& CooMatrix<A, T>::sortRowMajor()
{
  sortRowMajorByIterators<A>(rowIndices_.begin(), columnIndices_.begin(),
                             values_.begin(),
                             values_.size());
  
  return *this;
}

template<Architecture A, typename T>
inline void CooMatrix<A, T>::shrinkToFit()
{
  rowIndices_.shrink_to_fit();
  columnIndices_.shrink_to_fit();
  values_.shrink_to_fit();
}

template<Architecture A, typename T>
inline void CooMatrix<A, T>::reserve(uint64_t capacity)
{
  rowIndices_.reserve(capacity);
  columnIndices_.reserve(capacity);
  values_.reserve(capacity);
}

template<Architecture A, typename T>
inline void CooMatrix<A, T>::resize(uint64_t nValues)
{
  rowIndices_.resize(nValues);
  columnIndices_.resize(nValues);
  values_.resize(nValues);
}

template<Architecture A, typename T>
inline void CooMatrix<A, T>::save(const std::string& fileName) const
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
    putNumberIntoJSONStream(ofs, "nRows", nRows_);
    ofs << ",";
    putNumberIntoJSONStream(ofs, "nColumns", nColumns_);
    ofs << ",";
    putNumberIntoJSONStream(ofs, "nnz", values_.size());
    ofs << ",";
  }
  
  if (ofs.good()) {
    std::vector<NrBoxes> _rowIndices(rowIndices_.size());
    thrust::copy(rowIndices_.begin(), rowIndices_.end(), _rowIndices.begin());
    putNumberVectorIntoJSONStream(ofs, "rowIndices", _rowIndices);
    ofs << ",";
    _rowIndices.clear();
    _rowIndices.shrink_to_fit();
  }
  
  if (ofs.good()) {
    std::vector<NrBoxes> _columnIndices(columnIndices_.size());
    thrust::copy(columnIndices_.begin(), columnIndices_.end(), _columnIndices.begin());
    putNumberVectorIntoJSONStream(ofs, "columnIndices", _columnIndices);
    ofs << ",";
    _columnIndices.clear();
    _columnIndices.shrink_to_fit();
  }
  
  if (ofs.good()) {
    std::vector<T> _values(values_.size());
    thrust::copy(values_.begin(), values_.end(), _values.begin());
    putNumberVectorIntoJSONStream(ofs, "values", _values);
    ofs << ",";
    _values.clear();
    _values.shrink_to_fit();
  }
  
  if (ofs.good()) {
    ofs << "\n}";
  }
  
  if (! ofs.good()) {
    std::cout << "Warning: An error occurred during writing." << std::endl;
  }
  
  ofs.close();
}

template<Architecture A, typename T>
inline std::ostream& operator<<(std::ostream& os, const CooMatrix<A, T>& cooMatrix)
{
  os << "CooMatrix: Dimension: " << cooMatrix.nRows_ << " * " << cooMatrix.nColumns_ << std::endl;
  
  for (uint64_t i=0; i<cooMatrix.values_.size(); ++i) {
    os << "(" << cooMatrix.rowIndices_[i] << "," << cooMatrix.columnIndices_[i] << ")\t" 
       << cooMatrix.values_[i] << std::endl;
  }
  
  return os;
}

} // namespace b12
