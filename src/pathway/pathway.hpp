#ifndef __UASTAR_PATHWAY
#define __UASTAR_PATHWAY

#include "problem.hpp"
#include "pathway/input.hpp"
#include "pathway/input/threedim.hpp"

class CPUPathwaySolver;
class GPUPathwaySolver;

class Pathway : public Problem {
public:
    Pathway();
    ~Pathway();
    string problemName() const;
    void prepare();
    void cpuInitialize();
    void gpuInitialize();
    void cpuSolve();
    void gpuSolve();
    bool output();

    int sx() const;
    int sy() const;
    int sz() const;
    int ex() const;
    int ey() const;
    int ez() const;
    int size() const;
    int width() const;
    int height() const;
    int toID(int x, int y) const;
    void toXY(int id, int *x, int *y) const;
    bool inrange(int x, int y) const;
    const uint8_t *graph() const;

    int layer(void) const;
    int toID(const int x, const int y, const int z) const;
    void toXYZ(const int id, int *x, int *y, int *z) const;
    void toXYZ(const int id, int &x, int &y, int &z) const;
    bool inrange(const int x, const int y, const int z) const;
private:
    void printGraph(void);
    void generateGraph(PathwayInput &input);
    void generateGraph(ThreedimPathwayInput &input);
    void printSolution(const vector<int> &pathList,
                       const string filename) const;
    void plotSolution(const vector<int> &pathList,
                      const string filename) const;

    int m_sx;
    int m_sy;
    int m_sz;
    int m_ex;
    int m_ey;
    int m_ez;
    int m_size;
    int m_width;
    int m_height;
    int m_layer;
    string m_inputModule;
    vector<uint8_t> m_graph;
    CPUPathwaySolver *cpuSolver;
    GPUPathwaySolver *gpuSolver;

    bool cpuSolved;
    bool cpuSuccessful;
    float cpuOptimal;
    vector<int> cpuSolution;

    bool gpuSolved;
    bool gpuSuccessful;
    float gpuOptimal;
    vector<int> gpuSolution;
};

inline int Pathway::sx() const
{
    return m_sx;
}

inline int Pathway::sy() const
{
    return m_sy;
}

inline int Pathway::sz() const
{
    return m_sz;
}

inline int Pathway::ex() const
{
    return m_ex;
}

inline int Pathway::ey() const
{
    return m_ey;
}

inline int Pathway::ez() const
{
    return m_ez;
}

inline int Pathway::size() const
{
    return m_size;
}

inline int Pathway::width() const
{
    return m_width;
}

inline int Pathway::height() const
{
    return m_height;
}

inline int Pathway::toID(int x, int y) const
{
    return x * width() + y;
}


inline void Pathway::toXY(int id, int *x, int *y) const
{
    *x = id / width();
    *y = id % width();
}

inline bool Pathway::inrange(int x, int y) const
{
    return 0 <= x && x < height() && 0 <= y && y < width();
}
    

inline const uint8_t *Pathway::graph() const
{
    return m_graph.data();
}

inline int Pathway::layer(void) const {
  return m_layer;
}

inline int Pathway::toID(const int x, const int y, const int z) const
{
    return z * width() * height() + x * width() + y;
}

inline void Pathway::toXYZ(const int id, int &x, int &y, int &z) const
{
  int plane_size = width() * height();
  int bias = id % plane_size;
  x = bias / width();
  y = bias % width();
  z = id / plane_size;
}

inline void Pathway::toXYZ(const int id, int *x, int *y, int *z) const
{
  int bias = id / layer();
    *x = (id - bias) / width();
    *y = (id - bias) % width();
    *z = bias;
}

inline bool Pathway::inrange(const int x, const int y, const int z) const
{
    return 0 <= x && x < height() && 0 <= y && y < width() && \
                0 <= z && z < layer();
}

#endif
