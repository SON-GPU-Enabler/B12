
#include "math.h"
#include <cctype>
#include <cstdio>
#include <iomanip>
#include <limits>

#include "mathFunctions.h"


namespace b12 {

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt>::ExtendedUInt64_t() : hi_(0), lo_(0) {}

template<typename UInt>
template<typename UInt2>
__host__ __device__
inline ExtendedUInt64_t<UInt>::ExtendedUInt64_t(const ExtendedUInt64_t<UInt2>& eui) : hi_(eui.hi_), lo_(eui.lo_) {}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt>::ExtendedUInt64_t(UInt hi, uint64_t lo) : hi_(hi), lo_(lo) {}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt>::ExtendedUInt64_t(bool b) : hi_(0), lo_(b) {}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt>::ExtendedUInt64_t(int i) : hi_(i < 0 ? -1 : 0), lo_(i) {}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt>::ExtendedUInt64_t(uint32_t ui) : hi_(0), lo_(ui) {}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt>::ExtendedUInt64_t(uint64_t ui) : hi_(0), lo_(ui) {}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt>::ExtendedUInt64_t(float x)
    : hi_(ldexpreal(x, -64)), lo_(x - ldexpreal(float(hi_), 64)) {}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt>::ExtendedUInt64_t(double x)
    : hi_(ldexpreal(x, -64)), lo_(x - ldexpreal(double(hi_), 64)) {}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt>::operator bool() const
{
  return hi_ || lo_;
}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt>::operator int() const
{
  return int(lo_);
}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt>::operator uint32_t() const
{
  return uint32_t(lo_);
}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt>::operator uint64_t() const
{
  return lo_;
}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt>::operator float() const
{
  return ldexpreal(float(hi_), 64) + float(lo_);
}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt>::operator double() const
{
  return ldexpreal(double(hi_), 64) + double(lo_);
}

template<typename UInt>
template<typename UInt2>
__host__ __device__
inline ExtendedUInt64_t<UInt>& ExtendedUInt64_t<UInt>::operator=(const ExtendedUInt64_t<UInt2>& eui)
{
  hi_ = eui.hi_;
  lo_ = eui.lo_;
  return *this;
}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt> ExtendedUInt64_t<UInt>::operator<<(int i) const
{
  return ExtendedUInt64_t(
      (i >= std::numeric_limits<UInt>::digits ? UInt(0) : hi_ << i) | UInt(i >= 64 ? lo_ << (i - 64) : lo_ >> (64 - i)),
      i >= 64 ? uint64_t(0) : lo_ << i);
}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt> ExtendedUInt64_t<UInt>::operator>>(int i) const
{
  return ExtendedUInt64_t(i >= std::numeric_limits<UInt>::digits ? UInt(0) : hi_ >> i,
                          i >= 64 ? uint64_t(hi_) >> (i - 64) : (lo_ >> i) | (uint64_t(hi_) << (64 - i)));
}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt> ExtendedUInt64_t<UInt>::operator~() const
{
  return ExtendedUInt64_t(~hi_, ~lo_);
}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt> ExtendedUInt64_t<UInt>::operator&(const ExtendedUInt64_t<UInt>& eui) const
{
  return ExtendedUInt64_t(hi_ & eui.hi_, lo_ & eui.lo_);
}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt> ExtendedUInt64_t<UInt>::operator|(const ExtendedUInt64_t<UInt>& eui) const
{
  return ExtendedUInt64_t(hi_ | eui.hi_, lo_ | eui.lo_);
}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt> ExtendedUInt64_t<UInt>::operator^(const ExtendedUInt64_t<UInt>& eui) const
{
  return ExtendedUInt64_t(hi_ ^ eui.hi_, lo_ ^ eui.lo_);
}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt> ExtendedUInt64_t<UInt>::operator+(const ExtendedUInt64_t<UInt>& eui) const
{
  ExtendedUInt64_t res;
  res.lo_ = lo_ + eui.lo_;
  res.hi_ = hi_ + eui.hi_ + UInt(res.lo_ < eui.lo_);
  return res;
}
  
template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt> ExtendedUInt64_t<UInt>::operator-() const
{
  return ~(*this) + ExtendedUInt64_t(1);
}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt> ExtendedUInt64_t<UInt>::operator-(const ExtendedUInt64_t<UInt>& eui) const
{
  uint64_t diff = lo_ - eui.lo_;
  return ExtendedUInt64_t(hi_ - (eui.hi_ + UInt(diff > lo_)), diff);
}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt>& ExtendedUInt64_t<UInt>::operator<<=(int i)
{
  hi_ = (i >= std::numeric_limits<UInt>::digits ? UInt(0) : hi_ << i) | UInt(i >= 64 ? lo_ << (i - 64) : lo_ >> (64 - i));
  lo_ = i >= 64 ? uint64_t(0) : lo_ << i;
  return *this;
}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt>& ExtendedUInt64_t<UInt>::operator>>=(int i)
{
  hi_ = i >= std::numeric_limits<UInt>::digits ? UInt(0) : hi_ >> i;
  lo_ = i >= 64 ? uint64_t(hi_) >> (i - 64) : (lo_ >> i) | (uint64_t(hi_) << (64 - i));
  return *this;
}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt>& ExtendedUInt64_t<UInt>::operator&=(const ExtendedUInt64_t<UInt>& eui)
{
  hi_ &= eui.hi_;
  lo_ &= eui.lo_;
  return *this;
}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt>& ExtendedUInt64_t<UInt>::operator|=(const ExtendedUInt64_t<UInt>& eui)
{
  hi_ |= eui.hi_;
  lo_ |= eui.lo_;
  return *this;
}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt>& ExtendedUInt64_t<UInt>::operator^=(const ExtendedUInt64_t<UInt>& eui)
{
  hi_ ^= eui.hi_;
  lo_ ^= eui.lo_;
  return *this;
}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt>& ExtendedUInt64_t<UInt>::operator+=(const ExtendedUInt64_t<UInt>& eui)
{
  lo_ += eui.lo_;
  hi_ += eui.hi_ + UInt(lo_ < eui.lo_);
  return *this;
}

template<typename UInt>
__host__ __device__
inline ExtendedUInt64_t<UInt>& ExtendedUInt64_t<UInt>::operator-=(const ExtendedUInt64_t<UInt>& eui)
{
  uint64_t diff = lo_ - eui.lo_;
  hi_ -= eui.hi_ + UInt(diff > lo_);
  lo_ = diff;
  return *this;
}

template<typename UInt>
__host__ __device__
inline bool ExtendedUInt64_t<UInt>::operator==(const ExtendedUInt64_t<UInt>& eui) const
{
  return hi_ == eui.hi_ && lo_ == eui.lo_;
}

template<typename UInt>
__host__ __device__
inline bool ExtendedUInt64_t<UInt>::operator!=(const ExtendedUInt64_t<UInt>& eui) const
{
  return hi_ != eui.hi_ || lo_ != eui.lo_;
}

template<typename UInt>
__host__ __device__
inline bool ExtendedUInt64_t<UInt>::operator<(const ExtendedUInt64_t<UInt>& eui) const
{
  return hi_ < eui.hi_ || hi_ == eui.hi_ && lo_ < eui.lo_;
}

template<typename UInt>
__host__ __device__
inline bool ExtendedUInt64_t<UInt>::operator>(const ExtendedUInt64_t<UInt>& eui) const
{
  return hi_ > eui.hi_ || hi_ == eui.hi_ && lo_ > eui.lo_;
}

template<typename UInt>
__host__ __device__
inline bool ExtendedUInt64_t<UInt>::operator<=(const ExtendedUInt64_t<UInt>& eui) const
{
  return hi_ < eui.hi_ || hi_ == eui.hi_ && lo_ <= eui.lo_;
}

template<typename UInt>
__host__ __device__
inline bool ExtendedUInt64_t<UInt>::operator>=(const ExtendedUInt64_t<UInt>& eui) const
{
  return hi_ > eui.hi_ || hi_ == eui.hi_ && lo_ >= eui.lo_;
}

template<typename UInt>
__host__ __device__
inline bool ExtendedUInt64_t<UInt>::safeMAD(uint8_t base, uint8_t digit)
{
  bool res = digit < base;
  if (res) {
    if (base == 2) {
      // res = is first bit NOT set, i.e. left-shift is ok
      // if so, left-shift by 1 and add the digit (0 or 1)
      *this = (res = ! (hi_ & (UInt(1) << (std::numeric_limits<UInt>::digits - 1)))) == true
              ? (*this << 1) | ExtendedUInt64_t(digit)
              : ExtendedUInt64_t(-1);
    } else if (base == 8) {
      // res = are first three bits all NOT set, i.e. left-shift is ok
      // if so, left-shift by 3 and add the digit (0, ..., 7)
      *this = (res = ! (hi_ & (UInt(7) << (std::numeric_limits<UInt>::digits - 3)))) == true
              ? (*this << 3) | ExtendedUInt64_t(digit)
              : ExtendedUInt64_t(-1);
    } else if (base == 10) {
      // res = is multiplication by 10 ok, i.e. is object <= MAX/10, where MAX/10 == 0x1999...999
      // if so, multiply by 10
      *this = (res = ! (ExtendedUInt64_t(UInt(-1) / UInt(10), 0x9999999999999999) < *this)) == true
              ? (*this << 3) + (*this << 1)
              : ExtendedUInt64_t(-1);
      if (res) {
        // res = is addition ok
        // if so, add that sum
        ExtendedUInt64_t temp = *this + ExtendedUInt64_t(digit);
        *this = (res = ! (temp < *this)) == true ? temp : ExtendedUInt64_t(-1);
      }
    } else if (base == 16) {
      // res = are first four bits all NOT set, i.e. left-shift is ok
      // if so, left-shift by 4 and add the digit (0, ..., 9, a, ..., f)
      *this = (res = ! (hi_ & (UInt(15) << (std::numeric_limits<UInt>::digits - 4)))) == true
              ? (*this << 4) | ExtendedUInt64_t(digit) 
              : ExtendedUInt64_t(-1);
    }
  }
  return res;
}
  
template<typename UInt>
__host__ __device__
inline int countLeadingZeros(const ExtendedUInt64_t<UInt>& eui)
{
  if (eui.hi_) {
    return countLeadingZeros(uint64_t(eui.hi_)) - (64 - std::numeric_limits<UInt>::digits);
  } else {
    return countLeadingZeros(eui.lo_) + std::numeric_limits<UInt>::digits;
  }
}
  
template<typename UInt>
inline std::ostream& operator<<(std::ostream& os, const ExtendedUInt64_t<UInt>& eui)
{
  std::ios_base::fmtflags f(os.flags());
  os << std::hex << std::showbase;
  if (eui.hi_) {
    os << eui.hi_ << std::noshowbase << std::setw(16) << std::setfill('0');
  }
  os << eui.lo_;
  os.flags(f);
  return os;
}
  
template<typename UInt>
inline std::istream& operator>>(std::istream& is, ExtendedUInt64_t<UInt>& eui)
{
  if (is.good()) {
    eui = 0;
    std::streambuf * pbuf = is.rdbuf();
    // discard whitespaces
    while (isspace(pbuf->sgetc())) {
      pbuf->sbumpc();
    }
    // check if negative
    bool isNegative = false;
    if (pbuf->sgetc() == '+') {
      pbuf->sbumpc();
    } else if (pbuf->sgetc() == '-') {
      isNegative = true;
      pbuf->sbumpc();
    }
    // determien base
    uint8_t base = 10;
    if (pbuf->sgetc() == '0') {
      pbuf->sbumpc();
      char c = pbuf->sgetc();
      if (c == 'b' || c == 'B') {
        base = 2;
        pbuf->sbumpc();
      } if (c == 'x' || c == 'X') {
        base = 16;
        pbuf->sbumpc();
      } else {
        base = 8;
      }
    }
    uint8_t digit;
    int c;
    // iterate characters until EOF is reached or invalid character is found
    while ((c = pbuf->sgetc()) != EOF && (digit = getNumberFromChar(c, base)) != base) {
      pbuf->sbumpc();
      if (! eui.safeMAD(base, digit)) {
        isNegative = false;
        is.setstate(std::ios_base::failbit);
        break;
      }
    }
    if (c == EOF) {
      is.setstate(std::ios_base::eofbit);
    }
    if (digit == base && ! isspace(c)) {
      is.setstate(std::ios_base::failbit);
    }
    if (isNegative) {
      eui = -eui;
    }
  }
  return is;
}

__host__ __device__
inline uint8_t getNumberFromChar(char c, uint8_t base)
{
  int res = c - int('0');
  if (base <= 10) {
    if (res < 0 || base <= res) {
      res = base;
    }
  } else {
    if (res < 0 || 10 <= res) {
      res = c - int('a') + 10;
      if (res < 10 || base <= res) {
        res = c - int('A') + 10;
        if (res < 10 || base <= res) {
          res = base;
        }
      }
    }
  }
  return res;
}

} // namespace b12
