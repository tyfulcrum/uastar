#ifndef __CUSTOM_HPP_THREEDIM
#define __CUSTOM_HPP_THREEDIM

#include "utils.hpp"
// #include "pathway/input.hpp"

class ThreedimPathwayInput /* : public PathwayInput */ {
public:
    ThreedimPathwayInput(const int height, const int width, const int layer);
    ~ThreedimPathwayInput();
    void generate(uint8_t graph[]);
    void getStartPoint(int *x, int *y, int *z);
    void getEndPoint(int *x, int *y, int *z);

protected:
    int m_height;
    int m_width;
    int m_layer;
    int m_sx, m_sy, m_sz;
    int m_ex, m_ey, m_ez;
};

#endif /* end of include guard: __CUSTOM_HPP_JWMXQYT3 */
