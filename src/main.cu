#include <math.h>
#include <stdio.h>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <chrono>
#include <execution>
#include <cuda_profiler_api.h>
#include "ManagedAllocator.hpp"
#include "Operations.h"
#include "ReduceMaxIdx.h"
#include "ReduceMaxIdxOptimized.h"
#include "ReduceMaxIdxOptimizedShared.h"
#include "ReduceMaxIdxOptimizedBlocks.h"
#include "ReduceMaxIdxOptimizedWarp.h"
#include "ReduceMaxIdxOptimizedWarpShared.h"
#include "Process.h"

template<class T>
using ManagedVector = std::vector<T, ManagedAllocator<T>>;

template <typename T>
std::ostream& operator<< (std::ostream& out, const ManagedVector<T>& v) {
  if ( !v.empty() ) {
    out << '[';
    std::copy (v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b\b]";
  }
  return out;
}

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      //if (abort) exit(code);
   }
}

void checkGpuMem()
{
  size_t freeM, totalM, usedM;
  size_t freeT, totalT;
  cudaMemGetInfo(&freeT,&totalT);
  freeM =(unsigned int)freeT/1048576;
  totalM=(unsigned int)totalT/1048576;
  usedM=totalM-freeM;
  printf ( "mem free %ld MB\nmem total %ld MB\nmem used %ld MB\n",freeM,totalM,usedM);
}

int main()
{
    checkGpuMem();
    
    const int N = 256000;

    ManagedVector<float> input(N);
    ManagedVector<float> output(1024 / 32);
    ManagedVector<int> outputIdx(1024 / 32);

    default_random_engine e;
    uniform_real_distribution<> dis(0, 1); // rage 0 - 1

    generate(begin(input), end(input), [&](){ return dis(e); });
    
    cudaMemPrefetchAsync(input.data(), input.size() * sizeof(float), 0, 0);
    cudaMemPrefetchAsync(output.data(), output.size() * sizeof(float), 0, 0);
    cudaMemPrefetchAsync(outputIdx.data(), outputIdx.size() * sizeof(int), 0, 0);

    gpuErrchk(cudaDeviceSynchronize());
    
    int iterations = 1000;

    cout << endl;
    cout << "iterations count=" << iterations << endl;
    cout << "array size=" << N << endl;
    cout << endl;

    process(iterations, "gpu", [&]()
    {
        reduceMaxIdx<<<1, 1>>>(input.data(), N, output.data(), outputIdx.data());
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
    });
    output[0] = 0;

    process(iterations, "otimized gpu", [&]()
    {
        reduceMaxIdxOptimized << <1, 1024 >> > (input.data(), N, output.data(), outputIdx.data());
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
    });
    output[0] = 0;

    process(iterations, "otimized shared gpu", [&]()
    {
        reduceMaxIdxOptimizedShared << <1, 1024 >> >(input.data(), N, output.data(), outputIdx.data());
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
    });
    output[0] = 0;

    process(iterations, "otimized block gpu", [&]()
    {
        reduceMaxIdxOptimizedBlocks << <4, 1024 >> >(input.data(), N, output.data(), outputIdx.data());
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
    });
    output[0] = 0;   

    process(iterations, "warp otimized gpu", [&]()
    {
        reduceMaxIdxOptimizedWarp << <1, 1024 >> >(input.data(), N, output.data(), outputIdx.data());
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
    });
    output[0] = 0;

    process(iterations, "warp shared otimized gpu", [&]()
    {
        reduceMaxIdxOptimizedWarpShared << <1, 1024 >> > (input.data(), N, output.data(), outputIdx.data());
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
    });
    output[0] = 0;
    
    cin.get();

    return 0;   
}