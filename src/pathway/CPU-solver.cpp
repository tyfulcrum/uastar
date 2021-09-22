#include "pathway/CPU-solver.hpp"

#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <fmt/core.h>
#include <array>
#include <algorithm>

using namespace std;

struct node_t {
    int id;
    float dist;
    node_t *prev;
    node_t() = default;
    node_t(int id, float dist, node_t *prev)
        : id(id), dist(dist), prev(prev) { }
};

CPUPathwaySolver::CPUPathwaySolver(Pathway *pathway)
    : p(pathway)
{
    // pass
}

CPUPathwaySolver::~CPUPathwaySolver()
{
    for (auto &pair : globalList)
        delete pair.second;
    globalList.clear();
}

void CPUPathwaySolver::initialize()
{
    for (auto &pair : globalList)
        delete pair.second;
    globalList.clear();

    openList = decltype(openList)();
    closeList.clear();

    targetID = p->toID(p->ex(), p->ey(), p->ez());

    int startID = p->toID(p->sx(), p->sy(), p->sz());
    node_t *startNode = new node_t(startID, 0, nullptr);
    openList.push(make_pair(computeFValue(startNode), startNode));
    globalList[startNode->id] = startNode;
    /*
    fmt::print("startID: {}, targetID: {}", startID, targetID);
    int tx, ty, tz;
    p->toXYZ(targetID, tx, ty, tz);
    fmt::print("targetID: ({}, {}, {})\n", tx, ty, tz);
    */
}

bool CPUPathwaySolver::solve()
{
  const int DIM = 6;
  const array<int, DIM> DX = { 0,  0,  1,  1, 0, 0 };
  const array<int, DIM> DY = { 1, -1,  0, 0, 0, 0 };
  const array<int, DIM> DZ = { 0, 0, 0, 0, 1, -1 };
  const array<int, DIM> COST = { 1, 1, 1, 1, 1, 1 };
    while (!openList.empty()) {
        node_t *node;
        do {
            node = openList.top().second;
            openList.pop();
        } while (closeList.count(node->id));
        closeList.insert(node->id);

        if (node->id == targetID) {
            optimalNode = node;
            return true;
        }

        int x, y, z;
        p->toXYZ(node->id, x, y, z);
        dout << "(" << x << ", " << y << ")" << endl;
        // fmt::print("({}, {}, {}) ", x, y, z);
        for (int i = 0; i < DIM; ++i) {
            if (~p->graph()[node->id] & 1 << i)
                continue;
            int nx = x + DX[i];
            int ny = y + DY[i];
            int nz = z + DZ[i];
            if (p->inrange(nx, ny, nz)) {
                int nid = p->toID(nx, ny, nz);
                float dist = node->dist + COST[i];
                if (globalList.count(nid) == 0) {
                    node_t *nnode = new node_t(nid, dist, node);
                    globalList[nid] = nnode;
                    openList.push(make_pair(computeFValue(nnode), nnode));
                    dout << "\t(" << nx << ", " << ny << ") n " << computeFValue(nnode) << endl;
                } else {
                    node_t *onode = globalList[nid];
                    if (dist < onode->dist) {
                        onode->dist = dist;
                        onode->prev = node;
                        openList.push(make_pair(computeFValue(onode), onode));
                        dout << '\t' << nx << " " << ny << " u " << computeFValue(onode) << endl;
                    }
                }
            }
        }
    }
    return false;
}

void CPUPathwaySolver::getSolution(float *optimal, vector<int> *pathList)
{
    printf("\t\t\tNumber of nodes expanded: %d\n", (int)globalList.size());
    node_t *node = optimalNode;
    *optimal = node->dist;
    pathList->clear();
    while (node) {
        pathList->push_back(node->id);
        node = node->prev;
    }
    std::reverse(pathList->begin(), pathList->end());
}

float CPUPathwaySolver::computeFValue(node_t *node)
{
    int x, y, z;
    p->toXYZ(node->id, x, y, z);
    int dx = abs(x - p->ex());
    int dy = abs(y - p->ey());
    int dz = abs(z - p->ez());
    return (float) dx + dy + dz;
}
