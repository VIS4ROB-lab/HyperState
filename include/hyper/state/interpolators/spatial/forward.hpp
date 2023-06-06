/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/forward.hpp"
#include "hyper/variables/forward.hpp"

namespace hyper::state {

template <typename TGroup>
class SpatialInterpolator;

template <int N>
using RnInterpolator = SpatialInterpolator<variables::Rn<N>>;

using SU2Interpolator = SpatialInterpolator<variables::SU2>;

using SE3Interpolator = SpatialInterpolator<variables::SE3>;

}  // namespace hyper::state
