
#pragma once

#include <cstdint>
#include <iostream>


namespace b12 {

template<typename UInt>
class ExtendedUInt64_t
{
 public:
  
  template<typename UInt2>
  friend class ExtendedUInt64_t;
  
  __host__ __device__
  ExtendedUInt64_t();
  
  template<typename UInt2>
  __host__ __device__
  ExtendedUInt64_t(const ExtendedUInt64_t<UInt2>& eui);
  
  __host__ __device__
  ExtendedUInt64_t(UInt hi, uint64_t lo);
  
  __host__ __device__
  ExtendedUInt64_t(bool b);
  
  __host__ __device__
  ExtendedUInt64_t(int i);
  
  __host__ __device__
  ExtendedUInt64_t(uint32_t ui);
  
  __host__ __device__
  ExtendedUInt64_t(uint64_t ui);
  
  __host__ __device__
  ExtendedUInt64_t(float x);
  
  __host__ __device__
  ExtendedUInt64_t(double x);
  
  __host__ __device__
  operator bool() const;
  
  __host__ __device__
  operator int() const;
  
  __host__ __device__
  operator uint32_t() const;
  
  __host__ __device__
  operator uint64_t() const;
  
  __host__ __device__
  operator float() const;
  
  __host__ __device__
  operator double() const;
  
  template<typename UInt2>
  __host__ __device__
  ExtendedUInt64_t& operator=(const ExtendedUInt64_t<UInt2>& eui);
  
  __host__ __device__
  ExtendedUInt64_t operator<<(int i) const;
  
  __host__ __device__
  ExtendedUInt64_t operator>>(int i) const;
  
  __host__ __device__
  ExtendedUInt64_t operator~() const;
  
  __host__ __device__
  ExtendedUInt64_t operator&(const ExtendedUInt64_t& eui) const;
  
  __host__ __device__
  ExtendedUInt64_t operator|(const ExtendedUInt64_t& eui) const;
  
  __host__ __device__
  ExtendedUInt64_t operator^(const ExtendedUInt64_t& eui) const;
  
  __host__ __device__
  ExtendedUInt64_t operator+(const ExtendedUInt64_t& eui) const;
  
  __host__ __device__
  ExtendedUInt64_t operator-() const;
  
  __host__ __device__
  ExtendedUInt64_t operator-(const ExtendedUInt64_t& eui) const;
  
  __host__ __device__
  ExtendedUInt64_t& operator<<=(int i);
  
  __host__ __device__
  ExtendedUInt64_t& operator>>=(int i);
  
  __host__ __device__
  ExtendedUInt64_t& operator&=(const ExtendedUInt64_t& eui);
  
  __host__ __device__
  ExtendedUInt64_t& operator|=(const ExtendedUInt64_t& eui);
  
  __host__ __device__
  ExtendedUInt64_t& operator^=(const ExtendedUInt64_t& eui);
  
  __host__ __device__
  ExtendedUInt64_t& operator+=(const ExtendedUInt64_t& eui);
  
  __host__ __device__
  ExtendedUInt64_t& operator-=(const ExtendedUInt64_t& eui);
  
  __host__ __device__
  bool operator==(const ExtendedUInt64_t& eui) const;
  
  __host__ __device__
  bool operator!=(const ExtendedUInt64_t& eui) const;
  
  __host__ __device__
  bool operator<(const ExtendedUInt64_t& eui) const;
  
  __host__ __device__
  bool operator>(const ExtendedUInt64_t& eui) const;
  
  __host__ __device__
  bool operator<=(const ExtendedUInt64_t& eui) const;
  
  __host__ __device__
  bool operator>=(const ExtendedUInt64_t& eui) const;
  
  // multiplies the object with base and adds digit;
  // returns whether no overflow occurs;
  // if overflow would occur, the object is set to ExtendedUInt64_t(-1)
  __host__ __device__
  bool safeMAD(uint8_t base, uint8_t digit);
  
  template<typename UInt2>
  __host__ __device__
  friend int countLeadingZeros(const ExtendedUInt64_t<UInt2>& eui);
  
  template<typename UInt2>
  friend std::ostream& operator<<(std::ostream& os, const ExtendedUInt64_t<UInt2>& i);
  
  template<typename UInt2>
  friend std::istream& operator>>(std::istream& is, ExtendedUInt64_t<UInt2>& i);
  
 private:
  UInt hi_;
  uint64_t lo_;
};

__host__ __device__
uint8_t getNumberFromChar(char c, uint8_t base);

} // namespace b12


#include "ExtendedUInt64_t.hpp"
