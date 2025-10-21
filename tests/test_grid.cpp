
#include "grid.hpp"
#include <cassert>
int main() {
    Grid g(10,20,3,1.0,2.0);
    assert(g.dr == 0.1);
    assert(g.dz == 0.1);
    return 0;
}
