#pragma once

__global__ void reduceMaxIdx(const float* __restrict__ input, const int size, float* maxOut, int* maxIdxOut)
{

//
//  Here is the plan. Each block will execute a different channel. Inside each block, every threads will
//  more or less execute the find max and sum for each batch element. This is to make the reduce operation easier.

    float max = 0.0;
    int maxIdx = 0;

    for(int i = 0; i < size; i++)
    {
        if (fabs(input[i]) > max)
        {
            max = fabs(input[i]);
            maxIdx = i;
        }
    }

    maxOut[0] = max;
    maxIdxOut[0] = maxIdx;
}