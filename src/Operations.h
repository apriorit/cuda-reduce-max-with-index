#pragma once

__device__ void atomicMax(float* const address, const float value)
{
	if (*address >= value)
	{
		return;
	}

	int* const addressAsI = (int*)address;
	int old = *addressAsI, assumed;

	do 
	{
		assumed = old;
		if (__int_as_float(assumed) >= value)
		{
			break;
		}

		old = atomicCAS(addressAsI, assumed, __float_as_int(value));
	} while (assumed != old);
}

__inline__ __device__ float warpReduceMax(float val) 
{
    const unsigned int FULL_MASK = 0xffffffff;

    for (int mask = warpSize / 2; mask > 0; mask /= 2) 
    {
        val = max(__shfl_xor_sync(FULL_MASK, val, mask), val);
    }
    
    return val;
}

template<class T>
__inline__ __device__ float warpBroadcast(T val, int predicate) 
{
    const unsigned int FULL_MASK = 0xffffffff;

    unsigned int mask = __ballot_sync(FULL_MASK, predicate);

    int lane = 0;
    for (;!(mask & 1); ++lane)
    {
        mask >>= 1;
    }
    
    return __shfl_sync(FULL_MASK, val, lane);
}