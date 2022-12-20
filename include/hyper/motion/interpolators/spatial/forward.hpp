/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/definitions.hpp"
#include "hyper/motion/interpolators/temporal/forward.hpp"

namespace hyper {

template <typename TVariable>
class SpatialInterpolator;

struct SpatialInterpolatorQuery {
  // Definitions.
  using MotionQuery = StateQuery;
  using Layout = TemporalInterpolatorLayout<Eigen::Index>;
  using Weights = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  // Members.
  const MotionQuery& motion_query;
  const Layout& layout;
  const Scalar* const* inputs;
  const Weights& weights;
};

} // namespace hyper
