/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/definitions.hpp"
#include "hyper/motion/interpolators/temporal/forward.hpp"

namespace hyper {

template <typename TVariable>
class SpatialInterpolator;

struct PolicyQuery {
  // Definitions.
  using Layout = TemporalInterpolatorLayout<Eigen::Index>;
  using Inputs = Pointers<const Scalar>;
  using Weights = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  // Members.
  const Layout& layout;
  const Inputs& inputs;
  const Weights& weights;
};

} // namespace hyper
