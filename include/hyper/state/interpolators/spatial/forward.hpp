/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/forward.hpp"
#include "hyper/variables/forward.hpp"

namespace hyper::state {

template <typename TOutput, typename TInput = TOutput>
class SpatialInterpolator;

template <typename TScalar, int N>
using RnInterpolator = SpatialInterpolator<variables::Rn<TScalar, N>>;

template <typename TScalar>
using SU2Interpolator = SpatialInterpolator<variables::SU2<TScalar>>;

template <typename TScalar>
using SE3Interpolator = SpatialInterpolator<variables::SE3<TScalar>>;

}  // namespace hyper::state
