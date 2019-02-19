#pragma once

__global__ void reduceMaxIdxOptimizedBlocks(const float* __restrict__ input, const int size, float* maxOut, int* maxIdxOut)
{
    __shared__ float sharedMax;
    __shared__ int sharedMaxIdx;

    if (0 == threadIdx.x)
    {
        sharedMax = 0.f;
        sharedMaxIdx = 0;
    }

    __syncthreads();

    float localMax = 0.f;
    int localMaxIdx = 0;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x)
    {
        float val = input[i];

        if (localMax < abs(val))
        {
            localMax = abs(val);
            localMaxIdx = i;
        }
    }

    atomicMax(&sharedMax, localMax);

    __syncthreads();

    if (sharedMax == localMax)
    {
        sharedMaxIdx = localMaxIdx;
    }

    __syncthreads();

    if (0 == threadIdx.x)
    {
        maxOut[blockIdx.x] = sharedMax;
        maxIdxOut[blockIdx.x] = sharedMaxIdx;
    }
}