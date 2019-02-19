#pragma once
#include <chrono>

template<class T>
void process(int iterations, const std::string mode, T&& kernelLambda)
{
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; i++)
    {
        kernelLambda();
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << mode << " time=" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
    std::cout << std::endl;
}
