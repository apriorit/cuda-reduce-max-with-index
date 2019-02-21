#pragma once

__global__ void reduceMaxIdxOptimized(const float* __restrict__ input, const int size, float* maxOut, int* maxIdxOut)
{
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

    atomicMax(maxOut, localMax);

    __syncthreads();

    if (*maxOut == localMax)
    {
        *maxIdxOut = localMaxIdx;
    }
}