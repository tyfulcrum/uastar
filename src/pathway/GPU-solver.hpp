#ifndef __GPU_SOLVER_HPP_1LYUPTGF
#define __GPU_SOLVER_HPP_1LYUPTGF

#include "pathway/pathway.hpp"
#include "vec2.hpp"

const int OPEN_LIST_SIZE = 1000000;
const int NODE_LIST_SIZE = OPEN_LIST_SIZE * 10;

const int NUM_BLOCK  = 13 * 2;
const int NUM_THREAD = 192 * 2;
const int NUM_TOTAL = NUM_BLOCK * NUM_THREAD;

const int VALUE_PER_THREAD = 1;
const int NUM_VALUE = NUM_TOTAL * VALUE_PER_THREAD;

const int HEAP_CAPACITY = OPEN_LIST_SIZE / NUM_TOTAL;

class DeviceData;
class GPUPathwaySolver {
public:
    GPUPathwaySolver(Pathway *pathway);
    ~GPUPathwaySolver();
    void initialize();
    bool solve(float *optimal, vector<vec2> *solution);

private:
    bool isPrime(uint32_t number);
    vector<uint32_t> genRandomPrime(uint32_t maximum, int count);
    // Problem
    Pathway *p;
    DeviceData *d;
};

#endif /* end of include guard: __GPU_SOLVER_HPP_1LYUPTGF */
