
#pragma once
#include "state.hpp"

namespace timeint {
// SSP RK2 driver (scaffold)
double step(Fields& F, const RunConfig& cfg, double t, double dt);
}
