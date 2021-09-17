#include "pathway/input/threedim.hpp"

ThreedimPathwayInput::ThreedimPathwayInput(const int height, const int width, \
    const int layer)
    : m_height(height), m_width(width), m_layer(layer)
{
    // pass
}

ThreedimPathwayInput::~ThreedimPathwayInput()
{
    // pass
}

void ThreedimPathwayInput::generate(uint8_t graph[])
{
    cin >> m_sx >> m_sy >> m_sz;
    cin >> m_ex >> m_ey >> m_ez;

    uint8_t *buf = graph;
    for (int i = 0; i < m_layer; ++i) {
      for (int j = 0; j < m_height; ++j) {
        for (int k = 0; k < m_width; ++k) {
          int t; cin >> t;
          *buf++ = t ? 0xFF : 0;
        }
      }
      cout << endl << "======" << endl;
    }
}

void ThreedimPathwayInput::getStartPoint(int *x, int *y, int *z)
{
    *x = m_sx;
    *y = m_sy;
    *z = m_sz;
}

void ThreedimPathwayInput::getEndPoint(int *x, int *y, int *z)
{
    *x = m_ex;
    *y = m_ey;
    *z = m_ez;
}
