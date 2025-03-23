
#pragma once

#include <cstdint>
#include <iostream>


namespace b12 {

using Flags = uint8_t;

enum Flag : Flags
{
  NONE = 0x00,
  HIT  = 0x01,
  INS  = 0x02,
  EXPD = 0x04,
  SD   = 0x08,
  EQU  = 0x10,
  ENTR = 0x20,
  EXTR = 0x40,
  ALL  = 0xFF
};

__host__ __device__
bool areFlagsSet(const Flags& flags, const Flags& flagsToCheck);

__host__ __device__
bool isAnyFlagSet(const Flags& flags, const Flags& flagsToCheck);

__host__ __device__
Flags set(const Flags& flags, const Flags& flagsToSet);

__host__ __device__
Flags unset(const Flags& flags, const Flags& flagsToUnset);

__host__ __device__
Flags change(const Flags& flags, const Flags& flagsToUnset, const Flags& flagsToSet);

} // namespace b12


#include "Flags.hpp"
