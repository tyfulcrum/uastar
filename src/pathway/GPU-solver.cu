#include <vector>
#define NO_CPP11

#include <iostream>
#include <moderngpu.cuh>
#include <queue>

#include "pathway/GPU-solver.hpp"
#include "pathway/GPU-kernel.cuh"
#include "utils.hpp"
#include <fmt/core.h>

bool debug = false;

using namespace mgpu;

using thrust::raw_pointer_cast;

int div_up(int x, int y) { return (x-1) / y + 1; }

struct DeviceData {
    // store the structure of the grid graph
    MGPU_MEM(uint8_t) graph;

    // store open list + close list
    MGPU_MEM(node_t) nodes;
    MGPU_MEM(int) nodeSize;

    // hash table for `nodes'
    MGPU_MEM(uint32_t) hash;
    // define the modules of sub hash table (not required in pathway finding
    // MGPU_MEM(uint32_t) modules;

    // store open list
    MGPU_MEM(heap_t) openList;
    // store the size for each heap
    MGPU_MEM(int) heapSize;
    MGPU_MEM(int) heapBeginIndex;

    // element waiting to be sorted
    MGPU_MEM(sort_t) sortList;
    // value for sortList, representing the its parents
    MGPU_MEM(uint32_t) prevList;
    // size of the preceding array
    MGPU_MEM(int) sortListSize;

    MGPU_MEM(sort_t) sortList2;
    MGPU_MEM(uint32_t) prevList2;
    MGPU_MEM(int) sortListSize2;

    MGPU_MEM(heap_t) heapInsertList;
    MGPU_MEM(int) heapInsertSize;

    // current shortest distance (a float)
    MGPU_MEM(uint32_t) optimalDistance;
    // store the result return by the GPU
    MGPU_MEM(heap_t) optimalNodes;
    // store the size for optimalNodes
    MGPU_MEM(int) optimalNodesSize;

    MGPU_MEM(uint32_t) lastAddr;
    MGPU_MEM(uint32_t) answerList;
    MGPU_MEM(int) answerSize;

    ContextPtr context;
};


GPUPathwaySolver::GPUPathwaySolver()
{
    d = new DeviceData();
}

GPUPathwaySolver::~GPUPathwaySolver()
{
    // vector<node_t> nodes;
    // vector<uint32_t> hash;
    // d->nodes->ToHost(nodes, d->nodeSize->Value());
    // d->hash->ToHost(hash, p->size());
    // for (;;) {
    //     cout << "(x, y): ";
    //     int x, y;
    //     int px, py;
    //     cin >> x >> y;
    //     int nodeID = p->toID(x, y);
    //     int hashValue = hash[nodeID];
    //     int prevID = nodes[nodes[hashValue].prev].nodeID;
    //     p->toXY(prevID, &px, &py);
    //     std::cout << "fValue: " << nodes[hashValue].fValue << endl
    //               << "gValue: " << nodes[hashValue].gValue << endl
    //               << "prev: " << px << ", " << py << endl << endl;;
    // }
    delete d;

}

int GPUPathwaySolver::gpuKnows(int x, int y, int z) {
  int res = false;
  device_vector<int> resvec;
  resvec.push_back(0);
  auto vec_data_ptr = thrust::raw_pointer_cast(&rd.zDirs[0]);
  auto resvec_ptr = thrust::raw_pointer_cast(&resvec[0]);
  read_bool_vec<<<1, 1>>>(resvec_ptr, x, y, z);
  res = resvec[0];
  return res;
}
bool GPUPathwaySolver::isEx(int x, int y, int z, frDirEnum dir, 
    frDirEnum lastdir) {
  bool res = false;
  device_vector<bool> resvec;
  resvec.push_back(false);
  auto res_ptr = thrust::raw_pointer_cast(&resvec[0]);
  test_isex<<<1, 1>>>(res_ptr, x, y, z, dir, lastdir);
  res = resvec[0];
  return res;
}

bool GPUPathwaySolver::testhasEdge(int x, int y, int z, frDirEnum dir) {
  bool res = false;
  device_vector<bool> resvec;
  resvec.push_back(false);
  auto res_ptr = thrust::raw_pointer_cast(&resvec[0]);
  test_isSrc<<<1, 1>>>(res_ptr, x, y, z);
  res = resvec[0];
  return res;
}

frDirEnum GPUPathwaySolver::testDir(int x, int y, int z) {
  frDirEnum res = frDirEnum::UNKNOWN;
  device_vector<frDirEnum> resvec;
  resvec.push_back(res);
  auto res_ptr = thrust::raw_pointer_cast(&resvec[0]);
  test_Dir<<<1, 1>>>(res_ptr, x, y, z);
  res = resvec[0];
  return res;
}


void GPUPathwaySolver::test_reverse(frMIdx &x, frMIdx &y, frMIdx &z, frDirEnum &dir) {
  device_vector<frDirEnum> dir_vec;
  dir_vec.push_back(dir);
  device_vector<frMIdx> idx_vec;
  idx_vec.push_back(x);
  idx_vec.push_back(y);
  idx_vec.push_back(z);
  auto dir_ptr = thrust::raw_pointer_cast(&dir_vec[0]);
  auto x_ptr = thrust::raw_pointer_cast(&idx_vec[0]);
  auto y_ptr = x_ptr + 1; // thrust::raw_pointer_cast(&idx_vec[1]);
  auto z_ptr = y_ptr + 1; // thrust::raw_pointer_cast(&idx_vec[2]);
  test_cuReverse<<<1, 1>>>(x_ptr, y_ptr, z_ptr, dir_ptr, x, y, z);
  x = idx_vec[0];
  y = idx_vec[1];
  z = idx_vec[2];
  dir = dir_vec[0];
}

cuWavefrontGrid GPUPathwaySolver::test_expand(frDirEnum dir, cuWavefrontGrid &grid, 
    const FlexMazeIdx &dstMazeIdx1, const FlexMazeIdx &dstMazeIdx2, 
    const frPoint &centerPt) {
  /*
  device_vector<cuWavefrontGrid> currgrid_vec;
  currgrid_vec.push_back(grid);
  device_vector<FlexMazeIdx> d_idx;
  d_idx.push_back(dstMazeIdx1);
  d_idx.push_back(dstMazeIdx2);
  device_vector<frPoint> ctrPt_vec;
  ctrPt_vec.push_back(centerPt);
  auto grid_ptr = raw_pointer_cast(&currgrid_vec[0]);
  auto center_ptr = raw_pointer_cast(&ctrPt_vec[0]);
  auto src_ptr = raw_pointer_cast(&d_idx[0]);
  auto dst1_ptr = src_ptr + 1;
  auto dst2_ptr = src_ptr + 2;
  device_vector<cuWavefrontGrid> resvec;
  resvec.push_back(0);
  auto res_ptr = raw_pointer_cast(&resvec[0]);
  test_cuexpand<<<1, 1>>>(res_ptr, grid_ptr, dir, dst1_ptr, dst2_ptr, center_ptr);
  auto res = resvec[0];
  */
  auto res = cuWavefrontGrid(grid);
  return res;
}

frCost GPUPathwaySolver::test_npCost(frDirEnum dir, cuWavefrontGrid &grid) {
  device_vector<frCost> resvec;
  resvec.push_back(0);
  auto res_ptr = raw_pointer_cast(&resvec[0]);
  test_getNCost_obj<<<1, 1>>>(res_ptr, dir, grid);
  auto res = resvec[0];
  return res;
}

frCost GPUPathwaySolver::test_npCost(frDirEnum dir, 
int xIn, int yIn, int zIn, frCoord layerPathAreaIn, 
          frCoord vLengthXIn, frCoord vLengthYIn,
          bool prevViaUpIn, frCoord tLengthIn,
          frCoord distIn, frCost pathCostIn, frCost costIn, 
          unsigned int backTraceBuffer
   ) {

  /*
  device_vector<cuWavefrontGrid> cuGrid;
  cuGrid.push_back(grid);
  fmt::print("cuGrid push over!\n");
  auto grid_ptr = raw_pointer_cast(&cuGrid[0]);
  */
  device_vector<frCost> resvec;
  resvec.push_back(0);
  auto res_ptr = raw_pointer_cast(&resvec[0]);
  test_getNCost<<<1, 1>>>(res_ptr, dir, xIn, yIn, zIn, layerPathAreaIn, vLengthXIn, vLengthYIn, 
      prevViaUpIn, tLengthIn, distIn, pathCostIn, costIn, backTraceBuffer);
  auto res = resvec[0];
  return res;
}

int GPUPathwaySolver::test_estcost(FlexMazeIdx src, FlexMazeIdx dst1, FlexMazeIdx dst2, frDirEnum dir) {
    device_vector<int> resvec;
    device_vector<FlexMazeIdx> d_idx;
    d_idx.push_back(src);
    d_idx.push_back(dst1);
    d_idx.push_back(dst2);
    resvec.push_back(0);
    auto res_ptr = raw_pointer_cast(&resvec[0]);
    auto src_ptr = raw_pointer_cast(&d_idx[0]);
    auto dst1_ptr = src_ptr + 1;
    auto dst2_ptr = src_ptr + 2;
    dtest_estcost<<<1, 1>>>(res_ptr, src_ptr, dst1_ptr, dst2_ptr, dir);
    auto res = resvec[0];
    return res;
}

void GPUPathwaySolver::initialize(const vector<unsigned long long> &bits, 
    const bovec &prevDirs, const bovec &srcs, 
        const bovec &guides, const bovec &zDirs, 
        const ivec &xCoords, const ivec &yCoords, const ivec &zCoords,
        const ivec &zHeights, const vector<frUInt4> &path_widths, 
        frUInt4 ggDRCCost, frUInt4 ggMarkerCost, 
        const ivec &via2ViaForbOverlapLen, const ivec &via2viaForbLen, 
        const ivec &viaForbiTurnLen, 
        bool drWorker_ava, int DRIter, int ripupMode, 
        int p_viaFOLen_size, int p_viaFLen_size, int p_viaFTLen_size
        )
{
    cudaDeviceSynchronize();
    cudaDeviceReset();

    d->context = CreateCudaDevice(0);

    rd.bits = bits;
    rd.prevDirs = prevDirs;
    rd.srcs = srcs;
    rd.guides = guides;
    rd.zDirs = zDirs;
    rd.xCoords = xCoords;
    rd.yCoords = yCoords;
    rd.zCoords = zCoords;
    rd.zHeights= zHeights;
    rd.path_widths = path_widths;
    rd.via2ViaForbOverlapLen = via2ViaForbOverlapLen;
    rd.via2viaForbLen = via2viaForbLen;
    rd.viaForbiTurnLen = viaForbiTurnLen;

    int x = xCoords.size();
    int y = yCoords.size();
    int z = zCoords.size();
    int layer_num = path_widths.size();

    auto bits_ptr = raw_pointer_cast(&rd.bits[0]);
    auto prevDirs_ptr = raw_pointer_cast(&rd.prevDirs[0]);
    auto srcs_ptr = raw_pointer_cast(&rd.srcs[0]);
    auto guides_ptr = raw_pointer_cast(&rd.guides[0]);
    auto zdirs_ptr = raw_pointer_cast(&rd.zDirs[0]);
    auto xCoords_ptr = raw_pointer_cast(&rd.xCoords[0]);
    auto yCoords_ptr = raw_pointer_cast(&rd.yCoords[0]);
    auto zCoords_ptr = raw_pointer_cast(&rd.zCoords[0]);
    auto zHeights_ptr = raw_pointer_cast(&rd.zHeights[0]);
    auto path_widths_ptr = raw_pointer_cast(&rd.path_widths[0]);
    auto vfol_ptr = raw_pointer_cast(&rd.via2ViaForbOverlapLen[0]);
    auto v2vfl_ptr = raw_pointer_cast(&rd.via2viaForbLen[0]);
    auto vftl_ptr = raw_pointer_cast(&rd.viaForbiTurnLen[0]);


    initializeDevicePointers(bits_ptr, prevDirs_ptr, srcs_ptr, guides_ptr, zdirs_ptr,
        xCoords_ptr, yCoords_ptr, zCoords_ptr, zHeights_ptr, x, y, z,
        path_widths_ptr, layer_num, ggDRCCost, ggMarkerCost, 
        drWorker, DRIter, ripupMode, 
        p_viaFOLen_size, p_viaFLen_size, p_viaFTLen_size);

    /*
    initializeCUDAConstantMemory(
        p->height(), p->width(), p->layer(), p->ex(), p->ey(), p->ez(), 
        (uint32_t)p->toID(p->ex(), p->ey(), p->ez()));

    d->graph = d->context->Malloc<uint8_t>(p->graph(), p->size());

    d->nodes = d->context->Malloc<node_t>(NODE_LIST_SIZE);
    d->nodeSize = d->context->Fill<int>(1, 1);

    d->hash = d->context->Fill<uint32_t>(p->size(), UINT32_MAX);

    d->openList = d->context->Malloc<heap_t>(OPEN_LIST_SIZE);
    d->heapSize = d->context->Fill<int>(NUM_TOTAL, 0);
    d->heapBeginIndex = d->context->Fill<int>(1, 0);

    d->sortList = d->context->Malloc<sort_t>(NUM_VALUE * 8);
    d->prevList = d->context->Malloc<uint32_t>(NUM_VALUE * 8);
    d->sortList2 = d->context->Malloc<sort_t>(NUM_VALUE * 8);
    d->prevList2 = d->context->Malloc<uint32_t>(NUM_VALUE * 8);
    d->sortListSize = d->context->Fill<int>(1, 0);
    d->sortListSize2 = d->context->Fill<int>(1, 0);

    d->heapInsertList = d->context->Malloc<heap_t>(NUM_VALUE * 8);
    d->heapInsertSize = d->context->Fill<int>(1, 0);

    d->optimalDistance = d->context->Fill<uint32_t>(1, UINT32_MAX);
    d->optimalNodes = d->context->Malloc<heap_t>(NUM_TOTAL);
    d->optimalNodesSize = d->context->Fill<int>(1, 0);

    d->lastAddr = d->context->Malloc<uint32_t>(1);
    d->answerList = d->context->Malloc<uint32_t>(ANSWER_LIST_SIZE);
    d->answerSize = d->context->Fill<int>(1, 0);

    kInitialize<<<1, 1>>>(
        *d->nodes,
        *d->hash,
        *d->openList,
        *d->heapSize,
        p->sx(),
        p->sy(), 
        p->sz()
    );
    dout << "\t\tGPU Initialization finishes" << endl;
        */
}

bool GPUPathwaySolver::solve()
{
    std::priority_queue< heap_t, vector<heap_t>, std::greater<heap_t> > pq;

    for (int round = 0; ;++round) {
        if (DEBUG_CONDITION) {
            vector<int> heapSize;
            d->heapSize->ToHost(heapSize, NUM_TOTAL);
            printf("\t\t\t Heapsize: %d of %d\n", heapSize[0], HEAP_CAPACITY);
        }

        // printf("\t\tRound %d\n", round); fflush(stdout);
        dprintf("\t\tRound %d: kExtractExpand\n", round);
        kExtractExpand<
            NUM_BLOCK, NUM_THREAD, VALUE_PER_THREAD, HEAP_CAPACITY> <<<
            NUM_BLOCK, NUM_THREAD>>>(
                *d->nodes,

                *d->graph,

                *d->openList,
                *d->heapSize,

                *d->optimalDistance,
                *d->optimalNodes,
                *d->optimalNodesSize,

                *d->sortList,
                *d->prevList,
                *d->sortListSize,

                // reset them BTW
                *d->heapBeginIndex,
                *d->heapInsertSize
            );
#ifdef KERNEL_LOG
        cudaDeviceSynchronize();
#endif

        dprintf("\t\tRound %d: Fetch optimalNodesSize: ", round);
        int optimalNodesSize = d->optimalNodesSize->Value();
        dprintf("%d\n", optimalNodesSize);

        if (optimalNodesSize) {
            printf("\t\tRound %d: Found one solution\n", round);
            vector<heap_t> optimalNodes;
            d->optimalNodes->ToHost(optimalNodes, optimalNodesSize);

            uint32_t optimalDistance = d->optimalDistance->Value();
            dprintf("\t\tRound %d: Fetch optimalDistance: %.2f\n", round, reverseFlipFloat(optimalDistance));

            for (size_t i = 0; i != optimalNodes.size(); ++i) {
                dprintf("\t\t\t optimalNodes[%d]: %.3f\n", (int)i, optimalNodes[i].fValue);
                pq.push(optimalNodes[i]);
            }

            dprintf("\t\t\t pq.top(): %.3f\n", pq.top().fValue);
            if (flipFloat(pq.top().fValue) <= optimalDistance) {
                printf("\t\t\t Number of nodes expanded: %d\n", d->nodeSize->Value());
                m_optimalNodeAddr = pq.top().addr;
                m_optimalDistance = pq.top().fValue;
                dprintf("\t\t\t Optimal nodes address: %d\n", m_optimalNodeAddr);
                return true;
            }
        }

        dprintf("\t\tRound %d: Fetch sortListSize: ", round);
        int sortListSize = d->sortListSize->Value();
        dprintf("%d\n", sortListSize);
        // if (round % 2000 == 0) {
        //     printf("\t\tRound %d: Fetch sortListSize: %d\n", round, sortListSize);
        // }
        if (sortListSize == 0)
            return false;

        dprintf("\t\tRound %d: MergesortPairs\n", round);
        MergesortPairs(
            d->sortList->get(),
            d->prevList->get(),
            sortListSize,
            *d->context
        );

        dprintf("\t\tRound %d: kAssign\n", round);
        kAssign<NUM_THREAD><<<
            div_up(sortListSize, NUM_THREAD), NUM_THREAD>>> (
                *d->sortList,
                *d->prevList,
                sortListSize,

                *d->sortList2,
                *d->prevList2,
                *d->sortListSize2
            );
#ifdef KERNEL_LOG
        cudaDeviceSynchronize();
#endif

        dprintf("\t\tRound %d: Fetch sortListSize2: ", round);
        int sortListSize2 = d->sortListSize2->Value();
        dprintf("%d\n", sortListSize2);
        // if (round % 2000 == 0) {
        //     printf("\t\tRound %d: Fetch sortListSize2: %d\n", round, sortListSize2);
        // }

        dprintf("\t\tRound %d: kDeduplicate\n", round);
        // printf("\t\tRound %d: nodeSize: %d\n", round, d->nodeSize->Value());
        kDeduplicate<NUM_THREAD> <<<
            div_up(sortListSize2, NUM_THREAD), NUM_THREAD>>> (
                *d->nodes,
                *d->nodeSize,

                *d->hash,

                *d->sortList2,
                *d->prevList2,
                sortListSize2,

                *d->heapInsertList,
                *d->heapInsertSize
            );
        // printf("\t\tRound %d: nodeSize: %d\n", round, d->nodeSize->Value());
#ifdef KERNEL_LOG
        cudaDeviceSynchronize();
#endif

        dprintf("\t\tRound %d: kHeapInsert\n", round);
        kHeapInsert<
            NUM_BLOCK, NUM_THREAD, HEAP_CAPACITY> <<<
            NUM_BLOCK, NUM_THREAD>>> (
                *d->openList,
                *d->heapSize,
                *d->heapBeginIndex,

                *d->heapInsertList,
                *d->heapInsertSize,

                // reset them BTW
                *d->sortListSize,
                *d->sortListSize2,
                *d->optimalDistance,
                *d->optimalNodesSize
            );
#ifdef KERNEL_LOG
        cudaDeviceSynchronize();
#endif
        dprintf("\t\tRound %d: Finished\n\n", round);
    }
}

void GPUPathwaySolver::getSolution(float *optimal, vector<int> *pathList)
{
    d->lastAddr->FromHost(&m_optimalNodeAddr, 1);
    kFetchAnswer<<<1, 1>>>(
        *d->nodes,

        *d->lastAddr,

        *d->answerList,
        *d->answerSize
    );

    int answerSize = d->answerSize->Value();

    vector<uint32_t> answerList;
    d->answerList->ToHost(answerList, answerSize);

    *optimal = m_optimalDistance;
    pathList->clear();
    pathList->reserve(answerSize);
    for (int i = answerSize-1; i >= 0; --i) {
        pathList->push_back((int)answerList[i]);
    }

}

bool GPUPathwaySolver::isPrime(uint32_t number)
{
    uint32_t upper = sqrt(number) + 1;
    assert(upper < number);

    for (uint32_t i = 2; i != upper; ++i)
        if (number % i == 0)
            return false;
    return true;
}

vector<uint32_t> GPUPathwaySolver::genRandomPrime(uint32_t maximum, int count)
{
    vector<uint32_t> result;
    int prepare = 3 * count;

    uint32_t now = maximum;
    while (prepare) {
        if (isPrime(now))
            result.push_back(now);
        now--;
    }

    std::random_shuffle(result.begin(), result.end());
    result.erase(result.begin() + count, result.end());

    for (int i = 0; i < count; ++i)
        dout << result[i] << " ";
    dout << endl;

    return result;
}
