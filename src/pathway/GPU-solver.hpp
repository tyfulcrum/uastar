#ifndef __GPU_SOLVER_HPP_1LYUPTGF
#define __GPU_SOLVER_HPP_1LYUPTGF

// #include "pathway/pathway.hpp"
#include <thrust/device_vector.h>
#include <vector>
#include <frCoreLangTypes.h>

using std::vector;
using thrust::device_vector;
using bovec = vector<bool>;
using namespace coret;

const int OPEN_LIST_SIZE = 10000000;
const int NODE_LIST_SIZE = 150000000;
const int ANSWER_LIST_SIZE = 50000;

const int NUM_BLOCK  = 13 * 3;
const int NUM_THREAD = 192;
const int NUM_TOTAL = NUM_BLOCK * NUM_THREAD;

const int VALUE_PER_THREAD = 1;
const int NUM_VALUE = NUM_TOTAL * VALUE_PER_THREAD;

const int HEAP_CAPACITY = OPEN_LIST_SIZE / NUM_TOTAL;

class DeviceData;
struct RoutingData {
  device_vector<unsigned long long> bits;
  device_vector<bool> srcs;
  device_vector<bool> prevDirs;
  device_vector<bool> guides;
  device_vector<bool> zDirs;
};
class GPUPathwaySolver {
public:
    GPUPathwaySolver();
    ~GPUPathwaySolver();
    void initialize(const vector<unsigned long long> &bits, 
        const bovec &prevDirs, const bovec &srcs, 
        const bovec &guides, const bovec &zDirs, int x, int y, int z);
    bool solve();
    int gpuKnows(int x, int y, int z);
    bool testhasEdge(int x, int y, int z, frDirEnum dir);
    bool isEx(int x, int y, int z, frDirEnum dir);
    frDirEnum testDir(int x, int y, int z);
    void getSolution(float *optimal, vector<int> *pathList);

private:
    bool isPrime(uint32_t number);
    vector<uint32_t> genRandomPrime(uint32_t maximum, int count);
    // Problem
    // Pathway *p;
    DeviceData *d;
    RoutingData rd;
    uint32_t m_optimalNodeAddr;
    float m_optimalDistance;
    
};

#endif /* end of include guard: __GPU_SOLVER_HPP_1LYUPTGF */
