#include <vector>
#include <algorithm>
#include <execution>
#include "ManagedAllocator.hpp"
#include <random>
#include "process.h"

using namespace std;

template<class ExPol>
void cpuMaxIdx(ExPol pol, const vector<float>& input, float* maxOut, int* maxIdxOut)
{
    auto maxIt = max_element(pol, begin(input), end(input), [](float a, float b) { return fabs(a) < fabs(b); });
    *maxOut = *maxIt;
    *maxIdxOut = maxIt - begin(input);
}

int main()
{
    const int N = 256000;

    vector<float> input(N);
    vector<float> output(1024 / 32);
    vector<int> outputIdx(1024 / 32);

    default_random_engine e;
    uniform_real_distribution<> dis(0, 1); // rage 0 - 1

    generate(begin(input), end(input), [&]() { return dis(e); });

    int iterations = 1000;

    cout << endl;
    cout << "iterations count=" << iterations << endl;
    cout << "array size=" << N << endl;
    cout << endl;

    process(iterations, "cpu seq", [&]()
    {
        cpuMaxIdx(execution::seq, input, output.data(), outputIdx.data());
    });
    output[0] = 0;

    process(iterations, "cpu par", [&]()
    {
        cpuMaxIdx(execution::par, input, output.data(), outputIdx.data());
    });
    output[0] = 0;

    process(iterations, "cpu rap unseq", [&]()
    {
        cpuMaxIdx(execution::par_unseq, input, output.data(), outputIdx.data());
    });
    output[0] = 0;

    cin.get();

    return 0;
}