#ifndef __GPU_KERNEL_CUH_IUGANILK
#define __GPU_KERNEL_CUH_IUGANILK

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <cstring>
#include <moderngpu.cuh>

#include "utils.hpp"
#include <frCoreLangTypes.h>
#include <dr/FlexMazeTypes.h>
#include <db/infra/frPoint.h>
#include <db/infra/frOrient.h>
using namespace coret;
using frPoint = fr::frPoint;
using FlexMazeIdx = fr::FlexMazeIdx;

// Suppose we only use x dimension
#define THREAD_ID (threadIdx.x)
#define GLOBAL_ID (THREAD_ID + NT * blockIdx.x)
#define BLOCK_ID  (blockIdx.x)

#define cudaAssert(X) \
    if ( !(X) ) { \
        printf( "Thread %d:%d failed assert at %s:%d!\n", \
                blockIdx.x, threadIdx.x, __FILE__, __LINE__ ); \
        return; \
    }
// #define KERNEL_LOG

using namespace mgpu;

struct heap_t {
    float fValue;
    uint32_t addr;
};

__host__ __device__ bool operator<(const heap_t &a, const heap_t &b)
{
    return a.fValue < b.fValue;
}
__host__ __device__ bool operator>(const heap_t &a, const heap_t &b)
{
    return a.fValue > b.fValue;
}

struct node_t {
    uint32_t prev;
    float fValue;
    float gValue;
    uint32_t nodeID;
};

struct sort_t {
    uint32_t nodeID;
    float gValue;
};

__host__ __device__ bool operator<(const sort_t &a, const sort_t &b)
{
    if (a.nodeID != b.nodeID)
        return a.nodeID < b.nodeID;
    return a.gValue < b.gValue;
}

inline __host__ __device__ uint32_t flipFloat(float fl)
{
    union {
        float fl;
        uint32_t  u;
    } un;
    un.fl = fl;
    return un.u ^ ((un.u >> 31) | 0x80000000);
}

inline __host__ __device__ float reverseFlipFloat(uint32_t u)
{
    union {
        float f;
        uint32_t u;
    } un;
    un.u = u ^ ((~u >> 31) | 0x80000000);
    return un.f;
}

__constant__ int d_height;
__constant__ int d_width;
__constant__ int d_targetX;
__constant__ int d_targetY;
__constant__ uint32_t d_targetID;
__constant__ uint32_t d_modules[10];

__constant__ int d_layer;
__constant__ int d_targetZ;

__device__ unsigned long long *d_bits;
__device__ bool *d_prevDirs;
__device__ bool *d_srcs;
__device__ bool *d_guides;
__device__ bool *d_zDirs;
__device__ int *xCoords;
__device__ int *yCoords;
__device__ int *zCoords;
__device__ int *zHeights;

__constant__ int DRIter;
__constant__ int ripupMode;

__constant__ frUInt4 ggDRCCost;

/* Layer info */
__device__ frPrefRoutingDirEnum *pref_dirs;
__device__ int *pitch;
__constant__ int top_layer_num;

/* Tech info */
__device__ int ****via2ViaForbiddenOverlapLen;
__device__ int ****via2ViaForbiddenLen;
__constant__ int via2ViaForbiddenOverlapLen_size;
__constant__ int via2ViaForbiddenLen_size;


inline __device__ int xyzToID(const int x, const int y, const int z)
{
  int plane_size = d_width * d_height;
    return z * plane_size + x * d_width + y;
}

inline __device__ void idToXYZ(const uint32_t id, int &x, int &y, int &z) {
  int plane_size = d_width * d_height;
  int bias = id % plane_size;
  x = bias / d_width;
  y = bias % d_width;
  z = id / plane_size;
}

inline __device__ void idToXYZ(const uint32_t nodeID, int *x, int *y, int *z)
{
  u_int32_t bias = nodeID / d_layer;
    *x = (nodeID - bias) / d_width;
    *y = (nodeID - bias) % d_width;
    *z = bias;
}

inline __device__ void idToXY(uint32_t nodeID, int *x, int *y)
{
    *x = nodeID / d_width;
    *y = nodeID % d_width;
}

inline __device__ int xyToID(int x, int y)
{
    return x * d_width + y;
}

inline __device__ float computeHValue(int x, int y)
{
    int dx = abs(d_targetX - x);
    int dy = abs(d_targetY - y);
    return min(dx, dy)*SQRT2 + abs(dx-dy);
}

inline __device__ float computeHValue(uint32_t nodeID)
{
    int x, y;
    idToXY(nodeID, &x, &y);
    return computeHValue(x, y);
}

inline __device__ float computeHValue(const int x, const int y, const int z)
{
    int dx = abs(d_targetX - x);
    int dy = abs(d_targetY - y);
    int dz = abs(d_targetZ - z);
    return (float) dx + dy + dz;
}

inline __device__ float computeHValue3D(uint32_t nodeID)
{
    int x, y, z;
    idToXYZ(nodeID, x, y, z);
    return computeHValue(x, y, z);
}


inline __device__ float inrange(const int x, const int y, const int z)
{
    return 0 <= x && x < d_height && 0 <= y && y < d_width && \
                0 <= z && z < d_layer;
}

inline __device__ float inrange(int x, int y)
{
    return 0 <= x && x < d_height && 0 <= y && y < d_width;
}

inline __device__ bool getZDir(int in){
  return d_zDirs[in];
}
inline __device__ frMIdx getIdx(frMIdx xIdx, frMIdx yIdx, frMIdx zIdx) {
  return (getZDir(zIdx)) ? (xIdx + yIdx * d_height + zIdx * d_height * d_width): 
    (yIdx + xIdx * d_width  + zIdx * d_height  * d_width);
}

__device__ void correct(frMIdx &x, frMIdx &y, frMIdx &z, frDirEnum &dir) {
  switch (dir) {
    case frDirEnum::W:
      x--;
      dir = frDirEnum::E;
      break;
    case frDirEnum::S:
      y--;
      dir = frDirEnum::N;
      break;
    case frDirEnum::D:
      z--;
      dir = frDirEnum::U;
      break;
    default:
      ;
  }
  return;
}

inline __device__ bool isValid(frMIdx x, frMIdx y, frMIdx z) {
  if (x < 0 || y < 0 || z < 0 ||
      x >= (frMIdx)d_height || y >= (frMIdx)d_width || z >= (frMIdx)d_layer) {
    return false;
  } else {
    return true;
  }
}
inline __device__ bool getBit(frMIdx idx, frMIdx pos) {
  return (d_bits[idx] >> pos ) & 1;
}

__device__ bool hasEdge(frMIdx x, frMIdx y, frMIdx z, frDirEnum dir) {
  correct(x, y, z, dir);
  if (isValid(x, y, z)) {
    auto idx = getIdx(x, y, z);
    printf("GPU getBit: %d\n", getBit(idx, 1));
    switch (dir) {
      case frDirEnum::E:
        return getBit(idx, 0);
      case frDirEnum::N:
        return getBit(idx, 1);
      case frDirEnum::U:
        return getBit(idx, 2);
      default:
        return false;
    }
  } else {
    return false;
  }
}
__device__ void reverse(frMIdx &x, frMIdx &y, frMIdx &z, frDirEnum &dir) {
  switch (dir) {
    case frDirEnum::E:
      x++;
      dir = frDirEnum::W;
      break;
    case frDirEnum::S:
      y--;
      dir = frDirEnum::N;
      break;
    case frDirEnum::W:
      x--;
      dir = frDirEnum::E;
      break;
    case frDirEnum::N:
      y++;
      dir = frDirEnum::S;
      break;
    case frDirEnum::U:
      z++;
      dir = frDirEnum::D;
      break;
    case frDirEnum::D:
      z--;
      dir = frDirEnum::U;
      break;
    default:
      ;
  }
  return;
}

__device__ bool hasGuide(frMIdx x, frMIdx y, frMIdx z, frDirEnum dir) {
  reverse(x, y, z, dir);
  auto idx = getIdx(x, y, z);
  return d_guides[idx];
}
inline __device__ bool isSrc(frMIdx x, frMIdx y, frMIdx z) {
  return d_srcs[getIdx(x, y, z)];
}
inline __device__ frDirEnum getPrevAstarNodeDir(frMIdx x, frMIdx y, frMIdx z) {
  auto baseIdx = 3 * getIdx(x, y, z);
  return (frDirEnum)(((unsigned short)(d_prevDirs[baseIdx]    ) << 2) + 
      ((unsigned short)(d_prevDirs[baseIdx + 1]) << 1) + 
      ((unsigned short)(d_prevDirs[baseIdx + 2]) << 0));
}

__global__ void hasEdge_test(bool *res, int x, int y, int z, frDirEnum dir) {
  *res = hasGuide(x, y, z, dir);
}

__device__ int cuStrcmp(const char *s1, const char *s2) {
  while(*s1 && (*s1 == *s2))
  {
    s1++;
    s2++;
  }
  return *(const unsigned char*)s1 - *(const unsigned char*)s2;
}

__device__ bool isExpandable(int x, int y, int z, frDirEnum gdir) {
  //bool enableOutput = true;
  bool enableOutput = false;
  frDirEnum dir = frDirEnum::S;
  frMIdx gridX = x;
  frMIdx gridY = y;
  frMIdx gridZ = z;
  bool hg = hasEdge(gridX, gridY, gridZ, dir);
  printf("GPU hasEdge: %d\n", hg);
  char *s1 = "abcd";
  char *s2 = "abc";
  fr::FlexMazeIdx Idx(1, 2, 3);
  Idx.set(2, 3, 1);
  fr::frOrient(fr::frOrientEnum::frcMXR90);
  fr::frPoint frpoint;
  printf("GPU FlexMazeIdx: (%d, %d, %d)\n", Idx.x(), Idx.y(), Idx.z());
  /*
  if (enableOutput) {
    if (!hasEdge(gridX, gridY, gridZ, dir)) {
      cout <<"no edge@(" <<gridX <<", " <<gridY <<", " <<gridZ <<") " <<(int)dir <<endl;
    }
    if (hasEdge(gridX, gridY, gridZ, dir) && !hasGuide(gridX, gridY, gridZ, dir)) {
      cout <<"no guide@(" <<gridX <<", " <<gridY <<", " <<gridZ <<") " <<(int)dir <<endl;
    }
  }
  */
  reverse(gridX, gridY, gridZ, dir);
  if (!hg || 
      isSrc(gridX, gridY, gridZ) || 
      (getPrevAstarNodeDir(gridX, gridY, gridZ) != frDirEnum::UNKNOWN) || // comment out for non-buffer enablement
      gdir == dir) {
    return false;
  } else {
    return true;
  }
}

inline __device__ void getNextGrid(frMIdx &gridX, frMIdx &gridY, frMIdx &gridZ, const frDirEnum dir) {
  switch(dir) {
    case frDirEnum::E:
      ++gridX;
      break;
    case frDirEnum::S:
      --gridY;
      break;
    case frDirEnum::W:
      --gridX;
      break;
    case frDirEnum::N:
      ++gridY;
      break;
    case frDirEnum::U:
      ++gridZ;
      break;
    case frDirEnum::D:
      --gridZ;
      break;
    default:
      ;
  }
  return;
}

inline __device__ int cuMax(int x, int y) {
  int result = x > y ? x : y;
  return result;
}
inline __device__ frPoint& getPoint(frPoint &in, frMIdx x, frMIdx y) {
  in.set(xCoords[x], yCoords[y]);
  return in;
}

inline __device__ frCoord getZHeight(frMIdx in) {
  return zHeights[in];
}
inline __device__ int getTableEntryIdx(bool in1, bool in2, bool in3) {
  int retIdx = 0;
  if (in1) {
    retIdx += 1;
  }
  retIdx <<= 1;
  if (in2) {
    retIdx += 1;
  }
  retIdx <<= 1;
  if (in3) {
    retIdx += 1;
  }
  return retIdx;
}

inline __device__ bool isIncluded(const int **intervals, int len) {
  bool included = false;
  auto interval = *intervals;
  if (interval != nullptr) {
      if (interval[0] <= len && interval[1] >= len) {
        included = true;
    }
  }
  return included;
}

// forbidden length table related 
inline __device__ bool isVia2ViaForbiddenLen(int tableLayerIdx, bool isPrevDown, bool isCurrDown, bool isCurrDirX, frCoord len, bool isOverlap = false) {
  int tableEntryIdx = getTableEntryIdx(!isPrevDown, !isCurrDown, !isCurrDirX);
  if (isOverlap) {
    return isIncluded(via2ViaForbiddenOverlapLen[tableLayerIdx][tableEntryIdx], len);
  } else {
    return isIncluded(via2ViaForbiddenLen[tableLayerIdx][tableEntryIdx], len);
  }
}

__device__ frCost getEstCost(const FlexMazeIdx &src, const FlexMazeIdx &dstMazeIdx1,
                                const FlexMazeIdx &dstMazeIdx2, const frDirEnum &dir) {
  // bend cost
  int bendCnt = 0;
  int forbiddenPenalty = 0;
  frPoint srcPoint, dstPoint1, dstPoint2;
  getPoint(srcPoint, src.x(), src.y());
  getPoint(dstPoint1, dstMazeIdx1.x(), dstMazeIdx1.y());
  getPoint(dstPoint2, dstMazeIdx2.x(), dstMazeIdx2.y());
  frCoord minCostX = cuMax(cuMax(dstPoint1.x() - srcPoint.x(), srcPoint.x() - dstPoint2.x()), 0) * 1;
  frCoord minCostY = cuMax(cuMax(dstPoint1.y() - srcPoint.y(), srcPoint.y() - dstPoint2.y()), 0) * 1;
  frCoord minCostZ = cuMax(cuMax(getZHeight(dstMazeIdx1.z()) - getZHeight(src.z()), 
                       getZHeight(src.z()) - getZHeight(dstMazeIdx2.z())), 0) * 1;


  bendCnt += (minCostX && dir != frDirEnum::UNKNOWN && dir != frDirEnum::E && dir != frDirEnum::W) ? 1 : 0;
  bendCnt += (minCostY && dir != frDirEnum::UNKNOWN && dir != frDirEnum::S && dir != frDirEnum::N) ? 1 : 0;
  bendCnt += (minCostZ && dir != frDirEnum::UNKNOWN && dir != frDirEnum::U && dir != frDirEnum::D) ? 1 : 0;

  int gridX = src.x();
  int gridY = src.y();
  int gridZ = src.z();
  getNextGrid(gridX, gridY, gridZ, dir);
  frPoint nextPoint;
  getPoint(nextPoint, gridX, gridY);
/*
  // avoid propagating to location that will cause fobidden via spacing to boundary pin
  if (DBPROCESSNODE == "GF14_13M_3Mx_2Cx_4Kx_2Hx_2Gx_LB") {
    if (DRIter >= 30 && ripupMode == 0) {
      if (dstMazeIdx1 == dstMazeIdx2 && gridZ == dstMazeIdx1.z()) {
        auto layerNum = (gridZ + 1) * 2;
        bool isH = (pref_dirs[layerNum] == frPrefRoutingDirEnum::frcHorzPrefRoutingDir);
        if (isH) {
          auto gap = abs(nextPoint.y() - dstPoint1.y());
          if (gap &&
              (isVia2ViaForbiddenLen(gridZ, false, false, false, gap, false) || layerNum - 2 < BOTTOM_ROUTING_LAYER) &&
              (isVia2ViaForbiddenLen(gridZ, true, true, false, gap, false) || layerNum + 2 > top_layer_num)) {
            forbiddenPenalty = pitch[layerNum] * ggDRCCost * 20;
          }
        } else {
          auto gap = abs(nextPoint.x() - dstPoint1.x());
          if (gap &&
              (getDesign()->getTech()->isVia2ViaForbiddenLen(gridZ, false, false, true, gap, false) || layerNum - 2 < BOTTOM_ROUTING_LAYER) &&
              (getDesign()->getTech()->isVia2ViaForbiddenLen(gridZ, true, true, true, gap, false) || layerNum + 2 > getDesign()->getTech()->getTopLayerNum())) {
            forbiddenPenalty = pitch[layerNum] * ggDRCCost * 20;
          }
        }
      }
    }
  }
*/

  return (minCostX + minCostY + minCostZ + bendCnt + forbiddenPenalty);
}

__global__ void dtest_estcost(int *res, FlexMazeIdx *src, 
    FlexMazeIdx *dstMazeIdx1, FlexMazeIdx *dstMazeIdx2, frDirEnum dir) {
  *res = getEstCost(*src, *dstMazeIdx1, *dstMazeIdx2, dir);
}

__global__ void test_isex(bool *res, int x, int y, int z, frDirEnum dir) {
  *res = isExpandable(x, y, z, dir);
}

__global__ void test_Dir(frDirEnum *res, int x, int y, int z) {
  *res = getPrevAstarNodeDir(x, y, z);
}

__global__ void read_bool_vec(int *res, int x, int y, int z) {
  // printf("\n GPU vec[%d] = %d\n", in, d_zDirs[in]);
  *res = getIdx(x, y, z);
}

inline cudaError_t initializeDevicePointers(
    unsigned long long *bits, bool *prevDirs, bool *srcs, bool *guides, bool *zDirs, 
    int *d_xCoords, int *d_yCoords, int *d_zCoords, int *d_zHeights, int height, int width, int z
)
{
    cudaError_t ret = cudaSuccess;
    ret = cudaMemcpyToSymbol(d_bits, &bits, sizeof(unsigned long long*));
    ret = cudaMemcpyToSymbol(d_prevDirs, &prevDirs, sizeof(bool*));
    ret = cudaMemcpyToSymbol(d_srcs, &srcs, sizeof(bool*));
    ret = cudaMemcpyToSymbol(d_guides, &guides, sizeof(bool*));
    ret = cudaMemcpyToSymbol(d_zDirs, &zDirs, sizeof(bool*));
    ret = cudaMemcpyToSymbol(xCoords, &d_xCoords, sizeof(int*));
    ret = cudaMemcpyToSymbol(yCoords, &d_yCoords, sizeof(int*));
    ret = cudaMemcpyToSymbol(zCoords, &d_zCoords, sizeof(int*));
    ret = cudaMemcpyToSymbol(zHeights, &d_zHeights, sizeof(int*));
    ret = cudaMemcpyToSymbol(d_height, &height, sizeof(int));
    ret = cudaMemcpyToSymbol(d_width, &width, sizeof(int));
    ret = cudaMemcpyToSymbol(d_layer, &z, sizeof(int));
    return ret;
}

inline cudaError_t initializeCUDAConstantMemory(
    int height,
    int width,
    int layer, 
    int targetX,
    int targetY,
    int targetZ,
    uint32_t targetID
)
{
    cudaError_t ret = cudaSuccess;
    ret = cudaMemcpyToSymbol(d_height, &height, sizeof(int));
    ret = cudaMemcpyToSymbol(d_width, &width, sizeof(int));
    ret = cudaMemcpyToSymbol(d_layer, &layer, sizeof(int));
    ret = cudaMemcpyToSymbol(d_targetX, &targetX, sizeof(int));
    ret = cudaMemcpyToSymbol(d_targetY, &targetY, sizeof(int));
    ret = cudaMemcpyToSymbol(d_targetZ, &targetZ, sizeof(int));
    ret = cudaMemcpyToSymbol(d_targetID, &targetID, sizeof(uint32_t));
    return ret;
}

inline cudaError_t updateModules(const vector<uint32_t> &mvec)
{
    return cudaMemcpyToSymbol(
        d_modules, mvec.data(), sizeof(uint32_t) * mvec.size());
}


__global__ void kInitialize(
    node_t g_nodes[],
    uint32_t g_hash[],
    heap_t g_openList[],
    int g_heapSize[],
    int startX,
    int startY, 
    int startZ
)
{
    node_t node;
    node.fValue = computeHValue(startX, startY, startZ);
    node.gValue = 0;
    node.prev = UINT32_MAX;
    node.nodeID = xyzToID(startX, startY, startZ);

    heap_t heap;
    heap.fValue = node.fValue;
    heap.addr = 0;

    g_nodes[0] = node;
    g_openList[0] = heap;
    g_heapSize[0] = 1;
    g_hash[node.nodeID] = 0;
}

// NB: number of CUDA block
// NT: number of CUDA thread per CUDA block
// VT: value handled per thread
template<int NB, int NT, int VT, int HEAP_CAPACITY>
__global__ void kExtractExpand(
    // global nodes
    node_t g_nodes[],

    uint8_t g_graph[],

    // open list
    heap_t g_openList[],
    int g_heapSize[],

    // solution
    uint32_t *g_optimalDistance,
    heap_t g_optimalNodes[],
    int *g_optimalNodesSize,

    // output buffer
    sort_t g_sortList[],
    uint32_t g_prevList[],
    int *g_sortListSize,

    // cleanup
    int *g_heapBeginIndex,
    int *g_heapInsertSize
)
{
    __shared__ uint32_t s_optimalDistance;
    __shared__ int s_sortListSize;
    __shared__ int s_sortListBase;

    int gid = GLOBAL_ID;
    int tid = THREAD_ID;
    if (tid == 0) {
        s_optimalDistance = UINT32_MAX;
        s_sortListSize = 0;
        s_sortListBase = 0;
    }

    __syncthreads();

    heap_t *heap = g_openList + HEAP_CAPACITY * gid - 1;

    heap_t extracted[VT];
    int popCount = 0;
    int heapSize = g_heapSize[gid];

#pragma unroll
    for (int k = 0; k < VT; ++k) {
        if (heapSize == 0)
            break;

        extracted[k] = heap[1];
        popCount++;

#ifdef KERNEL_LOG
        int x, y;
        idToXY(g_nodes[extracted[k].addr].nodeID, &x, &y);
        printf("\t\t\t[%d]: Extract (%d, %d){%.2f} in [%d]\n",
               gid, x, y, extracted[k].fValue, extracted[k].addr);
#endif

        heap_t nowValue = heap[heapSize--];

        int now = 1;
        int next;
        while ((next = now*2) <= heapSize) {
            heap_t nextValue = heap[next];
            heap_t nextValue2 = heap[next+1];
            bool inc = (next+1 <= heapSize) && (nextValue2 < nextValue);
            if (inc) {
                ++next;
                nextValue = nextValue2;
            }

            if (nextValue < nowValue) {
                heap[now] = nextValue;
                now = next;
            } else
                break;
        }
        heap[now] = nowValue;

    }
    g_heapSize[gid] = heapSize;

    const int DIM = 6;

    int sortListCount = 0;
    sort_t sortList[VT*DIM];
    int prevList[VT*DIM];
    bool valid[VT*DIM];

    const int DX[DIM] = { 1, -1, 0,  0, 0, 0 };
    const int DY[DIM] = { 0,  0, 1, -1, 0, 0 };
    const int DZ[DIM] = { 0,  0, 0, 0, 1, -1 };
    const float COST[DIM] = { 1, 1, 1, 1, 1, 1 };

#pragma unroll
    for (int k = 0; k < VT; ++k) {
#pragma unroll
        for (int i = 0; i < DIM; ++i)
            valid[k*DIM + i] = false;

        if (k >= popCount)
            continue;
        atomicMin(&s_optimalDistance, flipFloat(extracted[k].fValue));
        node_t node = g_nodes[extracted[k].addr];
        if (extracted[k].fValue != node.fValue)
            continue;

        if (node.nodeID == d_targetID) {
            int index = atomicAdd(g_optimalNodesSize, 1);
            g_optimalNodes[index] = extracted[k];
#ifdef KERNEL_LOG
            printf("\t\t\t Saved answer {%.3f}\n", extracted[k].fValue);
#endif
            continue;
        }

        int x, y, z;
        idToXYZ(node.nodeID, x, y, z);
#pragma unroll
        for (int i = 0; i < DIM; ++i) {
            if (~g_graph[node.nodeID] & (1 << i))
                continue;

            int nx = x + DX[i];
            int ny = y + DY[i];
            int nz = z + DZ[i];
            int index = k*DIM + i;
            if (inrange(nx, ny, nz)) {
                uint32_t nodeID = xyzToID(nx, ny, nz);
#ifdef KERNEL_LOG
                int px, py, pz;
                idToXYZ(node.nodeID, px, py, pz);
                printf("\t\t\t[%d]: Expand (%d, %d) from (%d, %d)\n",
                       gid, nx, ny, px, py);
#endif
                sortList[index].nodeID = nodeID;
                sortList[index].gValue = node.gValue + COST[i];
                prevList[index] = extracted[k].addr;
                valid[index] = true;
                ++sortListCount;
            }
        }
    }

    int sortListIndex = atomicAdd(&s_sortListSize, sortListCount);
    __syncthreads();
    if (tid == 0) {
        s_sortListBase = atomicAdd(g_sortListSize, s_sortListSize);
    }
    __syncthreads();
    sortListIndex += s_sortListBase;

#pragma unroll
    for (int k = 0; k < VT*DIM; ++k)
        if (valid[k]) {
            g_sortList[sortListIndex] = sortList[k];
            g_prevList[sortListIndex] = prevList[k];
            sortListIndex++;
        }
    if (tid == 0)
        atomicMin(g_optimalDistance, s_optimalDistance);
    if (gid == 0) {
        int newHeapBeginIndex = *g_heapBeginIndex + *g_heapInsertSize;

        *g_heapBeginIndex = newHeapBeginIndex % (NB*NT);
        *g_heapInsertSize = 0;
    }
}

// Assume g_sortList is sorted
template<int NT>
__global__ void kAssign(
    sort_t g_sortList[],
    uint32_t g_prevList[],
    int sortListSize,

    sort_t g_sortList2[],
    uint32_t g_prevList2[],
    int *g_sortListSize2
)
{
    __shared__ uint32_t s_nodeIDList[NT+1];
    __shared__ uint32_t s_sortListCount2;
    __shared__ uint32_t s_sortListBase2;

    int tid = THREAD_ID;
    int gid = GLOBAL_ID;

    bool working = false;
    sort_t sort;
    uint32_t prev;

    if (tid == 0)
        s_sortListCount2 = 0;

    if (tid == 0 && gid != 0)
        s_nodeIDList[0] = g_sortList[gid - 1].nodeID;

    if (gid < sortListSize) {
        working = true;
        sort = g_sortList[gid];
        prev = g_prevList[gid];
        s_nodeIDList[tid+1] = sort.nodeID;
    }
    __syncthreads();

    working &= (gid == 0 || s_nodeIDList[tid] != s_nodeIDList[tid+1]);

    int index;
    if (working) {
        index = atomicAdd(&s_sortListCount2, 1);
    }

    __syncthreads();
    if (tid == 0) {
         s_sortListBase2 = atomicAdd(g_sortListSize2, s_sortListCount2);
    }
    __syncthreads();

    if (working) {
        g_sortList2[s_sortListBase2 + index] = sort;
        g_prevList2[s_sortListBase2 + index] = prev;

#ifdef KERNEL_LOG
        int x, y;
        idToXY(sort.nodeID, &x, &y);
        printf("\t\t\t[%d]: Assign (%d %d){%.2f} from %d\n",
               gid, x, y, sort.gValue, prev);
#endif
    }
}

template<int NT>
__global__ void kDeduplicate(
    // global nodes
    node_t g_nodes[],
    int *g_nodeSize,

    // hash table
    uint32_t g_hash[],

    sort_t g_sortList[],
    uint32_t g_prevList[],
    int sortListSize,

    heap_t g_heapInsertList[],
    int *g_heapInsertSize
)
{
    int tid = THREAD_ID;
    int gid = GLOBAL_ID;
    bool working = gid < sortListSize;

    __shared__ int s_nodeInsertCount;
    __shared__ int s_nodeInsertBase;

    __shared__ int s_heapInsertCount;
    __shared__ int s_heapInsertBase;

    if (tid == 0) {
        s_nodeInsertCount = 0;
        s_heapInsertCount = 0;
    }
    __syncthreads();

    node_t node;
    bool insert = true;
    bool found = true;
    uint32_t nodeIndex;
    uint32_t heapIndex;
    uint32_t addr;

    if (working) {
        node.nodeID = g_sortList[gid].nodeID;
        node.gValue = g_sortList[gid].gValue;
        node.prev   = g_prevList[gid];
        node.fValue = node.gValue + computeHValue3D(node.nodeID);

        // cudaAssert((int)node.nodeID >= 0);
        addr = g_hash[node.nodeID];
        found = (addr != UINT32_MAX);

        if (found) {
            if (node.fValue < g_nodes[addr].fValue) {
                g_nodes[addr] = node;
            } else {
                insert = false;
            }
        }

        if (!found) {
            nodeIndex = atomicAdd(&s_nodeInsertCount, 1);
        }
        if (insert) {
            heapIndex = atomicAdd(&s_heapInsertCount, 1);
        }
    }

    __syncthreads();
    if (tid == 0) {
        s_nodeInsertBase = atomicAdd(g_nodeSize, s_nodeInsertCount);
        s_heapInsertBase = atomicAdd(g_heapInsertSize, s_heapInsertCount);
    }
    __syncthreads();

    if (working && !found) {
        addr = s_nodeInsertBase + nodeIndex;
#ifdef KERNEL_LOG
        int x, y;
        idToXY(node.nodeID, &x, &y);
        printf("\t\t\t[%d]: Store (%d, %d) to [%d]\n", gid, x, y, addr);
#endif
        g_hash[node.nodeID] = addr;
        g_nodes[addr] = node;
    }
    if (working && insert) {
        uint32_t index = s_heapInsertBase + heapIndex;
        g_heapInsertList[index].fValue = node.fValue;
        g_heapInsertList[index].addr = addr;
    }
}

template<int NB, int NT, int HEAP_CAPACITY>
__global__ void kHeapInsert(
    // open list
    heap_t g_openList[],
    int g_heapSize[],
    int *g_heapBeginIndex,

    heap_t g_heapInsertList[],
    int *g_heapInsertSize,

    // cleanup variable
    int *sortListSize,
    int *sortListSize2,
    uint32_t *optimalDistance,
    int *optimalNodesSize
)
{
    int gid = GLOBAL_ID;

    int heapInsertSize = *g_heapInsertSize;
    int heapIndex = *g_heapBeginIndex + gid;
    if (heapIndex >= NB*NT)
        heapIndex -= NB*NT;

    int heapSize = g_heapSize[heapIndex];
    heap_t *heap = g_openList + HEAP_CAPACITY * heapIndex - 1;

    for (int i = gid; i < heapInsertSize; i += NB*NT) {
        heap_t value = g_heapInsertList[i];
        int now = ++heapSize;

#ifdef KERNEL_LOG
        printf("\t\t\t[%d]: Push [%d] to heap %d\n",
               gid, value.addr, heapIndex);
#endif
        while (now > 1) {
            int next = now / 2;
            heap_t nextValue = heap[next];
            if (value < nextValue) {
                heap[now] = nextValue;
                now = next;
            } else
                break;
        }
        heap[now] = value;
    }

    g_heapSize[heapIndex] = heapSize;
    if (gid == 0) {
        *sortListSize = 0;
        *sortListSize2 = 0;
        *optimalDistance = UINT32_MAX;
        *optimalNodesSize = 0;
    }
}

__global__ void kFetchAnswer(
    node_t *g_nodes,

    uint32_t *lastAddr,

    uint32_t answerList[],
    int *g_answerSize
)
{
    int count = 0;
    int addr = *lastAddr;

    while (addr != UINT32_MAX) {
#ifdef KERNEL_LOG
        int x, y;
        idToXY(g_nodes[addr].nodeID, &x, &y);
        printf("\t\t\t Address: %d (%d, %d)\n", addr, x, y);
#endif
        answerList[count++] = g_nodes[addr].nodeID;
        addr = g_nodes[addr].prev;
    }

    *g_answerSize = count;
}

#endif /* end of include guard: __GPU_KERNEL_CUH_IUGANILK */
