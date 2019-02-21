#pragma once

__global__ void reduceMaxIdxOptimizedWarpShared(const float* __restrict__ input, const int size, float* maxOut, int* maxIdxOut)
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

    for (int i = threadIdx.x; i < size; i += blockDim.x)
    {
        float val = input[i];

        if (localMax < abs(val))
        {
            localMax = abs(val);
            localMaxIdx = i;
        }
    }

    const float warpMax = warpReduceMax(localMax);

    const int warpMaxXY = warpBroadcast(localMaxIdx, warpMax == localMax);

    const int lane = threadIdx.x % warpSize;

    if (lane == 0)
    {
        atomicMax(&sharedMax, warpMax);
    }

    __syncthreads();

    if (lane == 0)
    {
        if (sharedMax == warpMax)
        {
            sharedMaxIdx = warpMaxXY;
        }
    }

    __syncthreads();

    if (0 == threadIdx.x)
    {
        *maxOut = sharedMax;
        *maxIdxOut = sharedMaxIdx;
    }
}