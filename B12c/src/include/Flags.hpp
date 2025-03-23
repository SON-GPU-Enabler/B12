
#include "math.h"

#include "mathFunctions.h"


namespace b12 {

__host__ __device__
inline bool areFlagsSet(const Flags& flags, const Flags& flagsToCheck)
{
  return flagsToCheck == (flags & flagsToCheck);
}

__host__ __device__
inline bool isAnyFlagSet(const Flags& flags, const Flags& flagsToCheck)
{
  return flags & flagsToCheck;
}

__host__ __device__
inline Flags change(const Flags& flags, const Flags& flagsToUnset, const Flags& flagsToSet)
{
  return areFlagsSet(flags, flagsToUnset) ? set(unset(flags, flagsToUnset), flagsToSet) : flags;
}

__host__ __device__
inline Flags set(const Flags& flags, const Flags& flagsToSet)
{
  return flags | flagsToSet;
}

__host__ __device__
inline Flags unset(const Flags& flags, const Flags& flagsToUnset)
{
  return flags & ~(flagsToUnset);
}

} // namespace b12
