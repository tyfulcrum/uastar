#ifndef __GPU_SOLVER_HPP_1LYUPTGF
#define __GPU_SOLVER_HPP_1LYUPTGF

// #include "pathway/pathway.hpp"
#include <thrust/device_vector.h>
#include <vector>
#include <utility>
#include <frCoreLangTypes.h>
#include <dr/FlexMazeTypes.h>
#include <db/infra/frPoint.h>

using std::vector;
using thrust::device_vector;
using bovec = vector<bool>;
using ivec = vector<int>;
using namespace coret;
using fr::FlexMazeIdx;
using coret::cuWavefrontGrid;
using fr::frPoint;

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
struct RoutingData;
struct forBiddenRange_t;

class GPUPathwaySolver {
public:
    GPUPathwaySolver();
    ~GPUPathwaySolver();
    void initialize(const vector<unsigned long long> &bits, 
        const bovec &prevDirs, const bovec &srcs, 
        const bovec &guides, const bovec &zDirs, 
        const ivec &xCoords, const ivec &yCoords, const ivec &zCoords,
        const ivec &zHeights, const vector<frUInt4> &path_widths, 
        frUInt4 ggDRCCost, frUInt4 ggMarkerCost,
        const ivec &via2ViaForbOverlapLen, const ivec &via2viaForbLen, 
        const ivec &viaForbiTurnLen, 
        bool drWorker_ava, int drIter, int ripupMode, 
        int p_viaFOLen_size, int p_viaFLen_size, int p_viaFTLen_size, 
        vector<vector<vector<std::pair<frCoord, frCoord>>>> const &Via2ViaForbiddenOverlapLen, 
        vector<vector<vector<std::pair<frCoord, frCoord>>>> const &Via2ViaForbiddenLen,
        vector<vector<vector<std::pair<frCoord, frCoord>>>> const &ViaForbiddenTurnLen, 
        vector<std::pair<frCoord, frCoord>> const &halfViaEncArea, 
        std::string DBPROCESSNODE, frLayerNum topLayerNum_p);
    bool solve();
    frCost test_estcost(FlexMazeIdx src, FlexMazeIdx dst1, FlexMazeIdx dst2, frDirEnum dir);
    frCoord dtest_half(frMIdx z, bool f);
    int gpuKnows(int x, int y, int z);
    bool testhasEdge(int x, int y, int z, frDirEnum dir);
    bool isEx(int x, int y, int z, frDirEnum dir, frDirEnum lastdir);
    void test_reverse(frMIdx &x, frMIdx &y, frMIdx &z, frDirEnum &dir);
    frDirEnum testDir(int x, int y, int z);
    void getSolution(float *optimal, vector<int> *pathList);
    void printDeviceOverlapInfo(void);
    cuWavefrontGrid test_expand(frDirEnum dir, cuWavefrontGrid &grid, 
       const FlexMazeIdx &dstMazeIdx1, const FlexMazeIdx &dstMazeIdx2, 
       const frPoint &centerPt);
    frCost test_npCost(frDirEnum dir, cuWavefrontGrid &grid);
    frCost test_npCost(frDirEnum dir, 
        int xIn, int yIn, int zIn, frCoord layerPathAreaIn, 
        frCoord vLengthXIn, frCoord vLengthYIn,
        bool prevViaUpIn, frCoord tLengthIn,
        frCoord distIn, frCost pathCostIn, frCost costIn, unsigned int backTraceBufferIn);

private:

    forBiddenRange_t vectorPairCpy(vector<std::pair<frCoord, frCoord>> const &hostData);
    void forBiddenRangesDataCpy(
        device_vector<forBiddenRange_t *> &dest, 
        vector<vector<vector<std::pair<frCoord, frCoord>>>> const &hostData);
    bool isPrime(uint32_t number);
    vector<uint32_t> genRandomPrime(uint32_t maximum, int count);
    // Problem
    // Pathway *p;
    DeviceData *d;
    RoutingData *rd;
    uint32_t m_optimalNodeAddr;
    float m_optimalDistance;
};

#endif /* end of include guard: __GPU_SOLVER_HPP_1LYUPTGF */
