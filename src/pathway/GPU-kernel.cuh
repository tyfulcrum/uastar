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
#include <climits>
#include <thrust/device_vector.h>
#include <thrust/pair.h>
using namespace coret;
using frPoint = fr::frPoint;
using FlexMazeIdx = fr::FlexMazeIdx;
#define GRIDGRAPHDRCCOSTSIZE 8
int const BOTTOM_ROUTING_LAYER = 2;

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

struct forBiddenRange_t {
  size_t size;
  thrust::pair<frCoord, frCoord> *data;
};

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

__constant__ bool drWorker;
__constant__ int DRIter;
__constant__ int ripupMode;

__constant__ frUInt4 ggDRCCost;
__constant__ frUInt4 ggMarkerCost;

/* Layer info */
__device__ frPrefRoutingDirEnum *pref_dirs;
__device__ int *pitch;
__device__ frUInt4 *path_width;

/* Tech info */
__device__ int *via2ViaForbiddenOverlapLen;
__device__ int *via2ViaForbiddenLen;
__device__ int *viaForbiddenTurnLen;
__constant__ int via_sizeX;
/* size Y */
__constant__ int viaFOLen_size;
__constant__ int viaFLen_size;
__constant__ int viaFTLen_size;

__constant__ int d_forBiddenRange_layerNum;
__device__ forBiddenRange_t **overlap_data_addr;
__device__ forBiddenRange_t **len_data_addr;
__device__ forBiddenRange_t **turnlen_data_addr;
__constant__ forBiddenRange_t halfViaEncArea;
__constant__ char *DBPROCESSNODE;
__constant__ frLayerNum topLayerNum;



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
    // printf("GPU getBit: %d\n", getBit(idx, 1));
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
  *res = hasEdge(x, y, z, dir);
}

__device__ int cuStrcmp(const char *s1, const char *s2) {
  while(*s1 && (*s1 == *s2))
  {
    s1++;
    s2++;
  }
  return *(const unsigned char*)s1 - *(const unsigned char*)s2;
}

__device__ bool isExpandable(int x, int y, int z, frDirEnum dir, 
    frDirEnum lastdir) {
  //bool enableOutput = true;
  bool enableOutput = false;
  frMIdx gridX = x;
  frMIdx gridY = y;
  frMIdx gridZ = z;
  bool hg = hasEdge(gridX, gridY, gridZ, dir);
  // printf("GPU hasEdge: %d\n", hg);
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
      lastdir == dir) {
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
inline __device__ int getTableEntryIdx(bool in1, bool in2) {
  int retIdx = 0;
  if (in1) {
    retIdx += 1;
  }
  retIdx <<= 1;
  if (in2) {
    retIdx += 1;
  }
  return retIdx;
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

inline __device__ bool isIncluded(
    forBiddenRange_t const &intervals, 
    frCoord len) {
  bool included = false;
  size_t const size = intervals.size;
  if (size > 0) {
    for (size_t i = 0; i < size; ++i) {
      auto const &interval = *(intervals.data + i);
      if (interval.first <= len && interval.second >= len) {
        included = true;
        break;
      }
    }
  }
  return included;
}

__device__ bool isVia2ViaForbiddenLen(int tableLayerIdx, bool isPrevDown, 
    bool isCurrDown, bool isCurrDirX, frCoord len, bool isOverlap = false) {
  int tableEntryIdx = getTableEntryIdx(!isPrevDown, !isCurrDown, !isCurrDirX);
  if (isOverlap) {
    return isIncluded(*(*(overlap_data_addr+tableLayerIdx) + tableEntryIdx), len);
  } else {
    return isIncluded(*(*(len_data_addr+tableLayerIdx) + tableEntryIdx), len);
  }
}

__device__ bool isViaForbiddenTurnLen(int tableLayerIdx, bool isDown, 
    bool isCurrDirX, frCoord len) {
  int tableEntryIdx = getTableEntryIdx(!isDown, !isCurrDirX);
  return isIncluded(*(*(turnlen_data_addr+tableLayerIdx) + tableEntryIdx), len);
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

  // printf("device est cost = %d\n", minCostX + minCostY + minCostZ + bendCnt);
  int gridX = src.x();
  int gridY = src.y();
  int gridZ = src.z();
  getNextGrid(gridX, gridY, gridZ, dir);
  frPoint nextPoint;
  getPoint(nextPoint, gridX, gridY);
  // printf("DB data from device: %s\n", DBPROCESSNODE);
  // avoid propagating to location that will cause fobidden via spacing to boundary pin
  if (cuStrcmp(DBPROCESSNODE, "GF14_13M_3Mx_2Cx_4Kx_2Hx_2Gx_LB") == 0) {
    if (DRIter >= 30 && ripupMode == 0) {
      if (dstMazeIdx1 == dstMazeIdx2 && gridZ == dstMazeIdx1.z()) {
        auto layerNum = (gridZ + 1) * 2;
        bool isH = (pref_dirs[layerNum] == frPrefRoutingDirEnum::frcHorzPrefRoutingDir);
        if (isH) {
          auto gap = abs(nextPoint.y() - dstPoint1.y());
          if (gap &&
              (isVia2ViaForbiddenLen(gridZ, false, false, false, gap, false) || layerNum - 2 < BOTTOM_ROUTING_LAYER) &&
              (isVia2ViaForbiddenLen(gridZ, true, true, false, gap, false) || layerNum + 2 > topLayerNum)) {
            forbiddenPenalty = pitch[layerNum] * ggDRCCost * 20;
          }
        } else {
          auto gap = abs(nextPoint.x() - dstPoint1.x());
          if (gap &&
              (isVia2ViaForbiddenLen(gridZ, false, false, true, gap, false) || layerNum - 2 < BOTTOM_ROUTING_LAYER) &&
              (isVia2ViaForbiddenLen(gridZ, true, true, true, gap, false) || layerNum + 2 > topLayerNum)) {
            forbiddenPenalty = pitch[layerNum] * ggDRCCost * 20;
          }
        }
      }
    }
  }

  auto const result = minCostX + minCostY + minCostZ + bendCnt + forbiddenPenalty;
  // printf("result from device: %u\n", result);
  return result;
}
inline __device__ bool hasGridCostE(frMIdx x, frMIdx y, frMIdx z) {
  return getBit(getIdx(x, y, z), 12);
}
    // unsafe access, no check
inline __device__ bool hasGridCostN(frMIdx x, frMIdx y, frMIdx z) {
  return getBit(getIdx(x, y, z), 13);
}
    // unsafe access, no check
inline __device__ bool hasGridCostU(frMIdx x, frMIdx y, frMIdx z) {
  return getBit(getIdx(x, y, z), 14);
}

inline __device__ bool hasGridCost(frMIdx x, frMIdx y, frMIdx z, frDirEnum dir) {
  bool sol = false;
  correct(x, y, z, dir);
  switch(dir) {
    case frDirEnum::E: 
      sol = hasGridCostE(x, y, z);
      break;
    case frDirEnum::N:
      sol = hasGridCostN(x, y, z);
      break;
    default: 
      sol = hasGridCostU(x, y, z);
  }
  return sol;
}
inline __device__ frUInt4 getBits(frMIdx idx, frMIdx pos, frUInt4 length) {
  auto tmp = d_bits[idx] & (((1ull << length) - 1) << pos); // mask
  return tmp >> pos;
}

__device__ void correctU(frMIdx &x, frMIdx &y, frMIdx &z, frDirEnum &dir) {
  switch (dir) {
    case frDirEnum::D:
      z--;
      dir = frDirEnum::U;
      break;
    default:
      ;
  }
  return;
}

__device__ bool hasDRCCost(frMIdx x, frMIdx y, frMIdx z, frDirEnum dir) {
  frUInt4 sol = 0;
  if (dir != frDirEnum::D && dir != frDirEnum::U) {
    reverse(x, y, z, dir);
    auto idx = getIdx(x, y, z);
    sol = (getBits(idx, 16, GRIDGRAPHDRCCOSTSIZE));
  } else {
    correctU(x, y, z, dir);
    auto idx = getIdx(x, y, z);
    sol = (getBits(idx, 24, GRIDGRAPHDRCCOSTSIZE));
  }
  return (sol);
}

__device__ bool hasMarkerCost(frMIdx x, frMIdx y, frMIdx z, frDirEnum dir) {
  frUInt4 sol = 0;
  // old
  if (dir != frDirEnum::D && dir != frDirEnum::U) {
    reverse(x, y, z, dir);
    auto idx = getIdx(x, y, z);
    sol += (getBits(idx, 32, GRIDGRAPHDRCCOSTSIZE));
  } else {
    correctU(x, y, z, dir);
    auto idx = getIdx(x, y, z);
    sol += (getBits(idx, 40, GRIDGRAPHDRCCOSTSIZE));
  }
  return (sol);
}

__device__ bool isOverrideShapeCost(frMIdx x, frMIdx y, frMIdx z, frDirEnum dir) {
  if (dir != frDirEnum::D && dir != frDirEnum::U) {
    return false;
  } else {
    correctU(x, y, z, dir);
    auto idx = getIdx(x, y, z);
    return getBit(idx, 11);
  }
}

__device__ bool hasShapeCost(frMIdx x, frMIdx y, frMIdx z, frDirEnum dir) {
  frUInt4 sol = 0;
  if (dir != frDirEnum::D && dir != frDirEnum::U) {
    reverse(x, y, z, dir);
    auto idx = getIdx(x, y, z);
    sol = (getBits(idx, 56, GRIDGRAPHDRCCOSTSIZE));
  } else {
    correctU(x, y, z, dir);
    auto idx = getIdx(x, y, z);
    sol = isOverrideShapeCost(x, y, z, dir) ? 0 : (getBits(idx, 48, GRIDGRAPHDRCCOSTSIZE));
  }
  return (sol);
}
__device__ bool isBlocked(frMIdx x, frMIdx y, frMIdx z, frDirEnum dir) {
  correct(x, y, z, dir);
  if (isValid(x, y, z)) {
    auto idx = getIdx(x, y, z);
    switch (dir) {
      case frDirEnum::E:
        return getBit(idx, 3);
      case frDirEnum::N:
        return getBit(idx, 4);
      case frDirEnum::U:
        return getBit(idx, 5);
      default:
        return false;
    }
  } else {
    return false;
  }
}

__device__ frCoord getEdgeLength(frMIdx x, frMIdx y, frMIdx z, frDirEnum dir) {
  frCoord sol = 0;
  correct(x, y, z, dir);
  //if (isValid(x, y, z, dir)) {
  switch (dir) {
    case frDirEnum::E:
      sol = xCoords[x+1] - xCoords[x];
      break;
    case frDirEnum::N:
      sol = yCoords[y+1] - yCoords[y];
      break;
    case frDirEnum::U:
      sol = zHeights[z+1] - zHeights[z];
      break;
    default:
      ;
  }
  //}
  return sol;
}

inline __device__    frLayerNum getLayerNum(frMIdx z) {
  return zCoords[z];
}

__device__ void getPrevGrid(frMIdx &gridX, frMIdx &gridY, frMIdx &gridZ, const frDirEnum dir) {
  switch(dir) {
    case frDirEnum::E:
      --gridX;
      break;
    case frDirEnum::S:
      ++gridY;
      break;
    case frDirEnum::W:
      ++gridX;
      break;
    case frDirEnum::N:
      --gridY;
      break;
    case frDirEnum::U:
      --gridZ;
      break;
    case frDirEnum::D:
      ++gridZ;
      break;
    default:
      ;
  }
  return;
}

__device__ frCost getNextPathCost(const cuWavefrontGrid &currGrid, const frDirEnum &dir) {
  // bool enableOutput = true;
  bool enableOutput = false;
  frMIdx gridX = currGrid.x();
  frMIdx gridY = currGrid.y();
  frMIdx gridZ = currGrid.z();
  // printf("GPU processing (%d, %d, %d)\n", gridX, gridY, gridZ);
  frCost nextPathCost = currGrid.getPathCost();
  // bending cost
  auto currDir = currGrid.getLastDir();
  auto lNum = getLayerNum(currGrid.z());
  auto pathWidth = path_width[lNum];

  if (currDir != dir && currDir != frDirEnum::UNKNOWN) {
    // original
    ++nextPathCost;
  }
  auto oldcost = nextPathCost;

  // via2viaForbiddenLen enablement
  if (dir == frDirEnum::U || dir == frDirEnum::D) {
    frCoord currVLengthX = 0;
    frCoord currVLengthY = 0;
    currGrid.getVLength(currVLengthX, currVLengthY);
    bool isCurrViaUp = (dir == frDirEnum::U);
    bool isForbiddenVia2Via = false;
    // check only y
    if (currVLengthX == 0 && currVLengthY > 0 && isVia2ViaForbiddenLen(gridZ, !(currGrid.isPrevViaUp()), !isCurrViaUp, false, currVLengthY, false)) {
      isForbiddenVia2Via = true;
      // check only x
    } else if (currVLengthX > 0 && currVLengthY == 0 && isVia2ViaForbiddenLen(gridZ, !(currGrid.isPrevViaUp()), !isCurrViaUp, true, currVLengthX, false)) {
      isForbiddenVia2Via = true;
      // check both x and y
    } else if (currVLengthX > 0 && currVLengthY > 0 && 
        (isVia2ViaForbiddenLen(gridZ, !(currGrid.isPrevViaUp()), !isCurrViaUp, false, currVLengthY) &&
         isVia2ViaForbiddenLen(gridZ, !(currGrid.isPrevViaUp()), !isCurrViaUp, true, currVLengthX))) {
      isForbiddenVia2Via = true;
    }

    if (isForbiddenVia2Via) {
      if (drWorker && DRIter >= 3) {
        nextPathCost += ggMarkerCost * getEdgeLength(gridX, gridY, gridZ, dir);
      } else {
        nextPathCost += ggDRCCost * getEdgeLength(gridX, gridY, gridZ, dir);
      }
    }
  }
  //printf("GPU ok with (%d, %d, %d)\n", gridX, gridY, gridZ);

  // via2turn forbidden len enablement
  frCoord tLength    = INT_MAX;
  frCoord tLengthDummy = 0;
  bool    isTLengthViaUp = false;
  bool    isForbiddenTLen = false;
  if (currDir != frDirEnum::UNKNOWN && currDir != dir) {
    // next dir is a via
    if (dir == frDirEnum::U || dir == frDirEnum::D) {
      isTLengthViaUp = (dir == frDirEnum::U);
      // if there was a turn before
      if (tLength != INT_MAX) {
        if (currDir == frDirEnum::W || currDir == frDirEnum::E) {
          tLength = currGrid.getTLength();
          if (isViaForbiddenTurnLen(gridZ, !isTLengthViaUp, true, tLength)) {
            isForbiddenTLen = true;
          }
        } else if (currDir == frDirEnum::S || currDir == frDirEnum::N) {
          tLength = currGrid.getTLength();
          if (isViaForbiddenTurnLen(gridZ, !isTLengthViaUp, false, tLength)) {
            isForbiddenTLen = true;
          }
        }
      }
      // curr is a planar turn
    } else {
      isTLengthViaUp = currGrid.isPrevViaUp();
      if (currDir == frDirEnum::W || currDir == frDirEnum::E) {
      // printf("(%d, %d, %d) Ping!\n", gridX, gridY, gridZ);
        currGrid.getVLength(tLength, tLengthDummy);
        /*
        auto vvv = isViaForbiddenTurnLen(gridZ, !isTLengthViaUp, true, tLength);
        printf("GPU isViaForbiddenTurnLen: %d\n", vvv);
        */
        if (isViaForbiddenTurnLen(gridZ, !isTLengthViaUp, true, tLength)) {
          isForbiddenTLen = true;
        }
      } else if (currDir == frDirEnum::S || currDir == frDirEnum::N) {
        currGrid.getVLength(tLengthDummy, tLength);
      // printf("(%d, %d, %d) Pong!\n", gridX, gridY, gridZ);
        /*
          auto vvv = isViaForbiddenTurnLen(gridZ, !isTLengthViaUp, false, tLength);
          printf("GPU isViaForbiddenTurnLen: %d\n", vvv);
          */
        if (isViaForbiddenTurnLen(gridZ, !isTLengthViaUp, false, tLength)) {
          isForbiddenTLen = true;
        }
      }
    }
    if (isForbiddenTLen) {
      if (drWorker && DRIter >= 3) {
        nextPathCost += ggDRCCost * getEdgeLength(gridX, gridY, gridZ, dir);
      } else {
        nextPathCost += ggMarkerCost * getEdgeLength(gridX, gridY, gridZ, dir);
      }
    }
  }

  bool gridCost   = hasGridCost(gridX, gridY, gridZ, dir);
  bool drcCost    = hasDRCCost(gridX, gridY, gridZ, dir);
  bool markerCost = hasMarkerCost(gridX, gridY, gridZ, dir);
  bool shapeCost  = hasShapeCost(gridX, gridY, gridZ, dir);
  bool blockCost  = isBlocked(gridX, gridY, gridZ, dir);
  bool guideCost  = hasGuide(gridX, gridY, gridZ, dir);

  const frUInt4 GRIDCOST        = 2;
  const frUInt4 SHAPECOST       = 8;
  const frUInt4 BLOCKCOST       = 32;
  const frUInt4 GUIDECOST       = 1; // disabled change getNextPathCost to enable
  // temporarily disable guideCost

  auto gridCostv = gridCost    ? GRIDCOST         * getEdgeLength(gridX, gridY, gridZ, dir) : 0;
  auto drcCostv = drcCost     ? ggDRCCost        * getEdgeLength(gridX, gridY, gridZ, dir) : 0;
  auto markerCostv = markerCost  ? ggMarkerCost     * getEdgeLength(gridX, gridY, gridZ, dir) : 0;
  auto shapeCostv = shapeCost   ? SHAPECOST        * getEdgeLength(gridX, gridY, gridZ, dir) : 0;
  auto blockCostv = blockCost   ? BLOCKCOST        * pathWidth * 20                          : 0;
  auto guideCostv =!guideCost ? GUIDECOST        * getEdgeLength(gridX, gridY, gridZ, dir) : 0;

  nextPathCost += getEdgeLength(gridX, gridY, gridZ, dir)
    + (gridCost   ? GRIDCOST         * getEdgeLength(gridX, gridY, gridZ, dir) : 0)
    + (drcCost    ? ggDRCCost        * getEdgeLength(gridX, gridY, gridZ, dir) : 0)
    + (markerCost ? ggMarkerCost     * getEdgeLength(gridX, gridY, gridZ, dir) : 0)
    + (shapeCost  ? SHAPECOST        * getEdgeLength(gridX, gridY, gridZ, dir) : 0)
    + (blockCost  ? BLOCKCOST        * pathWidth * 20                          : 0)
    + (!guideCost ? GUIDECOST        * getEdgeLength(gridX, gridY, gridZ, dir) : 0);
  // printf("Are you OK? \n");
  /*
  if (enableOutput) {
    cout <<"edge grid/shape/drc/marker/blk/length = " 
      <<hasGridCost(gridX, gridY, gridZ, dir)   <<"/"
      <<hasShapeCost(gridX, gridY, gridZ, dir)  <<"/"
      <<hasDRCCost(gridX, gridY, gridZ, dir)    <<"/"
      <<hasMarkerCost(gridX, gridY, gridZ, dir) <<"/"
      <<isBlocked(gridX, gridY, gridZ, dir) <<"/"
      <<getEdgeLength(gridX, gridY, gridZ, dir) <<endl;
  }
  */
  // printf("GPU Costs %u: old cost: %u\n", nextPathCost, oldcost);
  return nextPathCost;

}

__device__ FlexMazeIdx getTailIdx(const FlexMazeIdx &currIdx, const cuWavefrontGrid &currGrid) {
  constexpr auto WAVEFRONTBUFFERSIZE = 2;
  constexpr auto DIRBITSIZE = 3;
  int gridX = currIdx.x();
  int gridY = currIdx.y();
  int gridZ = currIdx.z();
  auto backTraceBuffer = currGrid.getBackTraceBuffer();
  for (int i = 0; i < WAVEFRONTBUFFERSIZE; ++i) {
    int currDirVal = backTraceBuffer - ((backTraceBuffer >> DIRBITSIZE) << DIRBITSIZE);
    frDirEnum currDir = static_cast<frDirEnum>(currDirVal);
    backTraceBuffer >>= DIRBITSIZE;
    getPrevGrid(gridX, gridY, gridZ, currDir);
  }
  return FlexMazeIdx(gridX, gridY, gridZ);
}


__device__ frCoord getHalfViaEncArea(frMIdx z, bool isLayer1) {
  // printf("First: %d, Second: %d\n", halfViaEncArea.data[z].first, 
  //     halfViaEncArea.data[z].second);
  return (isLayer1 ? halfViaEncArea.data[z].first: halfViaEncArea.data[z].second);
}

__device__ cuWavefrontGrid expand(cuWavefrontGrid &dest,  cuWavefrontGrid &currGrid, const frDirEnum &dir, 
                                      const FlexMazeIdx &dstMazeIdx1, const FlexMazeIdx &dstMazeIdx2,
                                      const frPoint &centerPt) {
  bool enableOutput = false;
  //bool enableOutput = true;
  frCost nextEstCost, nextPathCost;
  int gridX = currGrid.x();
  int gridY = currGrid.y();
  int gridZ = currGrid.z();

  getNextGrid(gridX, gridY, gridZ, dir);
  
  FlexMazeIdx nextIdx(gridX, gridY, gridZ);
  // get cost
  nextEstCost = getEstCost(nextIdx, dstMazeIdx1, dstMazeIdx2, dir);
  nextPathCost = getNextPathCost(currGrid, dir);  
  /*
  if (enableOutput) {
    std::cout << "  expanding from (" << currGrid.x() << ", " << currGrid.y() << ", " << currGrid.z() 
              << ") [pathCost / totalCost = " << currGrid.getPathCost() << " / " << currGrid.getCost() << "] to "
              << "(" << gridX << ", " << gridY << ", " << gridZ << ") [pathCost / totalCost = " 
              << nextPathCost << " / " << nextPathCost + nextEstCost << "]\n";
  }
  */
  auto lNum = getLayerNum(currGrid.z());
  auto pathWidth = path_width[lNum];
  frPoint currPt;
  getPoint(currPt, gridX, gridY);
  frCoord currDist = abs(currPt.x() - centerPt.x()) + abs(currPt.y() - centerPt.y());

  // vlength calculation
  frCoord currVLengthX = 0;
  frCoord currVLengthY = 0;
  currGrid.getVLength(currVLengthX, currVLengthY);
  auto nextVLengthX = currVLengthX;
  auto nextVLengthY = currVLengthY;
  bool nextIsPrevViaUp = currGrid.isPrevViaUp();
  if (dir == frDirEnum::U || dir == frDirEnum::D) {
    nextVLengthX = 0;
    nextVLengthY = 0;
    nextIsPrevViaUp = (dir == frDirEnum::D); // up via if current path goes down
  } else {
    if (currVLengthX != INT_MAX &&
        currVLengthY != INT_MAX) {
      if (dir == frDirEnum::W || dir == frDirEnum::E) {
        nextVLengthX += getEdgeLength(currGrid.x(), currGrid.y(), currGrid.z(), dir);
      } else { 
        nextVLengthY += getEdgeLength(currGrid.x(), currGrid.y(), currGrid.z(), dir);
      }
    }
  }
  
  // tlength calculation
  auto currTLength = currGrid.getTLength();
  auto nextTLength = currTLength;
  // if there was a turn, then add tlength
  if (currTLength != INT_MAX) {
    nextTLength += getEdgeLength(currGrid.x(), currGrid.y(), currGrid.z(), dir);
  }
  // if current is a turn, then reset tlength
  if (currGrid.getLastDir() != frDirEnum::UNKNOWN && currGrid.getLastDir() != dir) {
    nextTLength = getEdgeLength(currGrid.x(), currGrid.y(), currGrid.z(), dir);
  }
  // if current is a via, then reset tlength
  if (dir == frDirEnum::U || dir == frDirEnum::D) {
    nextTLength = INT_MAX;
  }

  cuWavefrontGrid nextWavefrontGrid(gridX, gridY, gridZ, 
                                      currGrid.getLayerPathArea() + getEdgeLength(currGrid.x(), currGrid.y(), currGrid.z(), dir) * pathWidth, 
                                      nextVLengthX, nextVLengthY, nextIsPrevViaUp,
                                      nextTLength,
                                      currDist,
                                      nextPathCost, nextPathCost + nextEstCost, currGrid.getBackTraceBuffer());
  if (dir == frDirEnum::U || dir == frDirEnum::D) {
    nextWavefrontGrid.resetLayerPathArea();
    nextWavefrontGrid.resetLength();
    if (dir == frDirEnum::U) {
      nextWavefrontGrid.setPrevViaUp(false);
    } else {
      nextWavefrontGrid.setPrevViaUp(true);
    }
    nextWavefrontGrid.addLayerPathArea((dir == frDirEnum::U) ? getHalfViaEncArea(currGrid.z(), false) : getHalfViaEncArea(gridZ, true));
  }
  // update wavefront buffer
  auto tailDir = nextWavefrontGrid.shiftAddBuffer(dir);
  // non-buffer enablement is faster for ripup all
  // commit grid prev direction if needed
  auto tailIdx = getTailIdx(nextIdx, nextWavefrontGrid);
  /*
  if (tailDir != frDirEnum::UNKNOWN) {
    if (getPrevAstarNodeDir(tailIdx.x(), tailIdx.y(), tailIdx.z()) == frDirEnum::UNKNOWN ||
        getPrevAstarNodeDir(tailIdx.x(), tailIdx.y(), tailIdx.z()) == tailDir) {
      setPrevAstarNodeDir(tailIdx.x(), tailIdx.y(), tailIdx.z(), tailDir);
      // TODO: wavefront.push(nextWavefrontGrid);
      if (enableOutput) {
        std::cout << "    commit (" << tailIdx.x() << ", " << tailIdx.y() << ", " << tailIdx.z() << ") prev accessing dir = " << (int)tailDir << "\n";
      }
    }
  } else {  
    // TODO:  add to wavefront
    // wavefront.push(nextWavefrontGrid);
  }
  */

  return nextWavefrontGrid;
}


__global__ void test_getNCost_obj(frCost *res, frDirEnum dir, 
    cuWavefrontGrid grid) {
  *res = getNextPathCost(grid, dir);
}

__global__ void test_getNCost(frCost *res, frDirEnum dir, 
int xIn, int yIn, int zIn, frCoord layerPathAreaIn, 
          frCoord vLengthXIn, frCoord vLengthYIn,
          bool prevViaUpIn, frCoord tLengthIn,
          frCoord distIn, frCost pathCostIn, frCost costIn, 
          unsigned int backTraceBufferIn
    ) {
  cuWavefrontGrid grid(xIn, yIn, zIn, layerPathAreaIn, vLengthXIn, vLengthYIn, 
      prevViaUpIn, tLengthIn, distIn, pathCostIn, costIn, backTraceBufferIn);
  auto const result = getNextPathCost(grid, dir);
  *res = result;
}

__global__ void dtest_estcost(frCost *res, FlexMazeIdx src, 
    FlexMazeIdx dstMazeIdx1, FlexMazeIdx dstMazeIdx2, frDirEnum dir) {
  auto const result = getEstCost(src, dstMazeIdx1, dstMazeIdx2, dir);
  // printf("Result from device: %u\n", result);
  *res = result;
}

__global__ void test_isex(bool *res, int x, int y, int z, frDirEnum dir, 
    frDirEnum lastdir) {
  *res = isExpandable(x, y, z, dir, lastdir);
}

__global__ void test_halfviaenc(frCoord *res, frMIdx z, bool isLayer1) {
  *res = getHalfViaEncArea(z, isLayer1);
}

__global__ void test_Dir(frDirEnum *res, int x, int y, int z) {
  *res = getPrevAstarNodeDir(x, y, z);
}

__global__ void test_cuReverse(frMIdx *x, frMIdx *y, frMIdx *z, frDirEnum *dir, 
    frMIdx ix, frMIdx iy, frMIdx iz) {
  auto rx = ix;
  auto ry = iy;
  auto rz = iz;
  reverse(rx, ry, rz, *dir);
  *x = rx;
  *y = ry;
  *z = rz;
}

__global__ void test_hasEdge(bool *res, 
    frMIdx x, frMIdx y, frMIdx z, frDirEnum dir) {
  *res = hasEdge(x, y, z, dir);
}

__global__ void test_isSrc(bool *res, frMIdx x, frMIdx y, frMIdx z) {
  *res = isSrc(x, y, z);
}

__global__ void read_bool_vec(int *res, int x, int y, int z) {
  // printf("\n GPU vec[%d] = %d\n", in, d_zDirs[in]);
  *res = getIdx(x, y, z);
}
__global__ void test_cuexpand(cuWavefrontGrid *res, cuWavefrontGrid *grid, frDirEnum dir, 
    const FlexMazeIdx *dstMazeIdx1, const FlexMazeIdx *dstMazeIdx2, 
    const frPoint *centerPt) {
  *res = expand(*grid, *grid, dir, *dstMazeIdx1, *dstMazeIdx2, *centerPt);
}

__global__ void test_print_device_overlap_info(void) {
  int layerNum = d_forBiddenRange_layerNum;
  auto fbRanges_ptr = overlap_data_addr;
  printf("============Device Data==============\n");
  printf("Layer Num: %d\n", layerNum);
  for (int i = 0; i < layerNum; ++i, ++fbRanges_ptr) {
    // printf("Layer No.%d:\n", i);
    auto fbRanges = *fbRanges_ptr;
    for (int j = 0; j < 8; ++j) {
      // printf("Direction: %d:\n", j);
      auto p = fbRanges[j];
      if (p.size > 0) {
        for (size_t k = 0; k < p.size; ++k) {
          auto range = p.data[k];
          printf("[%d][%d](%d, %d) ", i, j,  range.first, range.second);
        }
        printf("\n");
      }
    }
  }
  printf("============Device Data END==============\n");
}

inline cudaError_t initializeDevicePointers(
    unsigned long long *bits, bool *prevDirs, bool *srcs, bool *guides, bool *zDirs, 
    int *d_xCoords, int *d_yCoords, int *d_zCoords, int *d_zHeights, int height, 
    int width, int z, frUInt4 *path_widths, frUInt4 p_ggDRCCost, 
    frUInt4 p_ggMarkerCost, bool drWorker_ava, int p_DRIter, int p_ripupMode, 
    int p_viaFOLen_size, int p_viaFLen_size, int p_viaFTLen_size, 
    forBiddenRange_t **p_overlap_addr_ptr, forBiddenRange_t **p_len_addr_ptr, 
    forBiddenRange_t **p_turnlen_addr_ptr
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
    ret = cudaMemcpyToSymbol(path_width, &path_widths, sizeof(frUInt4*));
    ret = cudaMemcpyToSymbol(ggDRCCost, &p_ggDRCCost, sizeof(frUInt4));
    ret = cudaMemcpyToSymbol(ggMarkerCost, &p_ggMarkerCost, sizeof(frUInt4));
    ret = cudaMemcpyToSymbol(drWorker, &drWorker_ava, sizeof(bool));
    ret = cudaMemcpyToSymbol(DRIter, &p_DRIter, sizeof(int));
    ret = cudaMemcpyToSymbol(ripupMode, &p_ripupMode, sizeof(int));
    ret = cudaMemcpyToSymbol(viaFOLen_size, &p_viaFOLen_size, sizeof(int));
    ret = cudaMemcpyToSymbol(viaFLen_size, &p_viaFLen_size, sizeof(int));
    ret = cudaMemcpyToSymbol(viaFTLen_size, &p_viaFTLen_size, sizeof(int));
    ret = cudaMemcpyToSymbol(overlap_data_addr, &p_overlap_addr_ptr, sizeof(forBiddenRange_t **));
    ret = cudaMemcpyToSymbol(len_data_addr, &p_len_addr_ptr, sizeof(forBiddenRange_t **));
    ret = cudaMemcpyToSymbol(turnlen_data_addr, &p_turnlen_addr_ptr, sizeof(forBiddenRange_t **));
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
