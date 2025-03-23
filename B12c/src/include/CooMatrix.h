
#pragma once

#include <cstdint>
#include <iostream>
#include <string>

#include <thrust/functional.h>
#include <thrust/tuple.h>

#include "ThrustSystem.h"
#include "TypeDefinitions.h"


namespace b12 {

typedef thrust::tuple<NrBoxes, NrBoxes> MatrixIndex;


// compresses a (sorted) index vector such that the corresponding matrix can be seen in CSC or CSR format;
// dim is the number of rows or columns of the matrix
template<Architecture A>
void compressIndexVector(typename ThrustSystem<A>::Vector<NrBoxes>& indices, NrBoxes dim);

// inverse operation of compressIndexVector
// dim is the number of rows or columns of the matrix
template<Architecture A>
void extendCompressedIndexVector(typename ThrustSystem<A>::Vector<NrBoxes>& indices);

// ordering of coo matrix indices, here:
// smaller column precedes bigger column,
// for equal columns: smaller row precedes bigger row
struct ColumnMajorOrderingFunctor : public thrust::binary_function<MatrixIndex, MatrixIndex, bool>
{
  __host__ __device__
  bool operator()(MatrixIndex mi1, MatrixIndex mi2) const;
};

// ordering of coo matrix indices, here:
// smaller row precedes bigger row,
// for equal rows: smaller column precedes bigger column
struct RowMajorOrderingFunctor : public thrust::binary_function<MatrixIndex, MatrixIndex, bool>
{
  __host__ __device__
  bool operator()(MatrixIndex mi1, MatrixIndex mi2) const;
};

template<Architecture A, typename IndexIterator, typename ValuesIterator>
void sortColumnMajorByIterators(IndexIterator row_begin, IndexIterator column_begin,
                                ValuesIterator values_begin, NrPoints n);

template<Architecture A, typename IndexIterator, typename ValuesIterator>
void sortRowMajorByIterators(IndexIterator row_begin, IndexIterator column_begin,
                             ValuesIterator values_begin, NrPoints n);

template<Architecture A, typename T>
struct CooMatrix
{
  /// default constructor, but reserves capacity matrix elements
  CooMatrix(NrBoxes nRows = 0, NrBoxes nColumns = 0, uint64_t capacity = 0);
  
  template<Architecture A2, typename T2>
  CooMatrix(const CooMatrix<A2, T2>& cooMatrix);
  
  template<typename T2>
  CooMatrix(const typename ThrustSystem<A>::Vector<NrBoxes>& rowIndices,
            const typename ThrustSystem<A>::Vector<NrBoxes>& columnIndices,
            const typename ThrustSystem<A>::Vector<T2>& values,
            NrBoxes nRows, NrBoxes nColumns, uint64_t capacity = 0);
  
  template<typename NrBoxesIterator, typename ValuesIterator>
  CooMatrix(NrBoxesIterator row_begin, NrBoxesIterator column_begin, ValuesIterator values_begin,
            uint64_t nValues, NrBoxes nRows, NrBoxes nColumns, uint64_t capacity = 0);
  
  /// loads the matrix that was saved into file 'fileName'
  CooMatrix(const std::string& fileName);
  
  template<Architecture A2, typename T2>
  CooMatrix<A, T>& operator=(const CooMatrix<A2, T2>& cooMatrix);
  
  /// returns true if both matrices are the same (independent of orderings)
  template<Architecture A2, typename T2>
  bool operator==(const CooMatrix<A2, T2>& cooMatrix) const;
  
  /// checks if the matrix is sorted in column-major format
  bool isSortedColumnMajor() const;
  
  /// checks if the matrix is sorted in row-major format
  bool isSortedRowMajor() const;
  
  /// transforms the matrix to its symmetric or antisymmetric part;
  /// preserves major ordering if pre-existing, otherwise column-major format is used
  void makeSymmetric(bool makeAntisymmetric = false);
  
  /// sorts the matrix in column-major format
  CooMatrix<A, T>& sortColumnMajor();
  
  /// sorts the matrix in row-major format
  CooMatrix<A, T>& sortRowMajor();
  
  /// shrinks the capacity of the vectors to exactly fit their elements
  void shrinkToFit();
  
  /// allocates enough memory for vectors
  void reserve(uint64_t capacity);
  
  /// resizes the vectors
  void resize(uint64_t nValues);
  
  /// stores the object on hard-disk in file 'fileName' overwriting previously existing data in that file;
  /// if fileName does not have the suffix ".json", that will be appended
  void save(const std::string& fileName) const;
  
  // prints the matrix
  template<Architecture A2, typename T2>
  friend std::ostream& operator<<(std::ostream& os, const CooMatrix<A2, T2>& cooMatrix);
  
  typename ThrustSystem<A>::Vector<NrBoxes> rowIndices_, columnIndices_;
  typename ThrustSystem<A>::Vector<T> values_;
  NrBoxes nRows_, nColumns_;
};

} // namespace b12


#include "CooMatrix.hpp"
