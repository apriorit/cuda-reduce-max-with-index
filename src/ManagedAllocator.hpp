#pragma once

#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>

template<class T>
class ManagedAllocator
{
  public:
    using value_type = T;

    ManagedAllocator() {}

    template<class U>
    ManagedAllocator(const ManagedAllocator<U>&) {}
  
    value_type* allocate(size_t n)
    {
      value_type* result = nullptr;
  
      cudaError_t error = cudaMallocManaged(&result, n*sizeof(T), cudaMemAttachGlobal);
  
      if(error != cudaSuccess)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "ManagedAllocator::allocate(): cudaMallocManaged");
      }
  
      return result;
    }
  
    void deallocate(value_type* ptr, size_t)
    {
      cudaError_t error = cudaFree(ptr);
  
      if(error != cudaSuccess)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "ManagedAllocator::deallocate(): cudaFree");
      }
    }
};

template<class T1, class T2>
bool operator==(const ManagedAllocator<T1>&, const ManagedAllocator<T2>&)
{
  return true;
}

template<class T1, class T2>
bool operator!=(const ManagedAllocator<T1>& lhs, const ManagedAllocator<T2>& rhs)
{
  return !(lhs == rhs);
}

