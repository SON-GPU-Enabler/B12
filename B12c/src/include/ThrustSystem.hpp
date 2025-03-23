
#include <iostream>

#include <thrust/memory.h>


namespace b12 {

template<>
inline bool Memory<HOST>::isOnHost()
{
  return true;
}

template<>
inline int Memory<HOST>::printDevice()
{
  return 0;
}

template<>
inline uint64_t Memory<HOST>::getFreeBytes(uint64_t initFreeBytes)
{
  uint64_t temp = 0;
  
  while (temp == 0) {
    thrust::host_system_tag host_sys;
    
    // try to temporarily allocate initFreeBytes elements of type char with thrust::get_temporary_buffer;
    // if ptr_and_size.second == 0, allocation failed, otherwise initFreeBytes Bytes can be allocated
    auto ptr_and_size = thrust::get_temporary_buffer<char>(host_sys, initFreeBytes);
    
    temp = ptr_and_size.second;
    
    // deallocate storage with thrust::return_temporary_buffer
    thrust::return_temporary_buffer(host_sys, ptr_and_size.first, ptr_and_size.second);
    
    // reduce the value since algorithms seem to need more space in the peak on HOST than on DEVICE, whyever
    initFreeBytes /= 4;
    initFreeBytes *= 3;
  }
  
  return initFreeBytes;
}

template<>
inline bool Memory<DEVICE>::isOnHost()
{
  return false;
}

template<>
inline int Memory<DEVICE>::printDevice()
{
  int nrDevice;
  cudaGetDevice(&nrDevice);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, nrDevice);
  std::cout << prop.name << " (PCI Bus ID: " 
            << prop.pciBusID << ") is the current device for the calling host thread." << std::endl;
  return nrDevice;
}

template<>
inline uint64_t Memory<DEVICE>::getFreeBytes(uint64_t dummy)
{
  uint64_t freeBytes, totalBytes;
  cudaMemGetInfo(&freeBytes, &totalBytes);
  return freeBytes;
}

} // namespace b12
