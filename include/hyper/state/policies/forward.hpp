/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/interpolators/forward.hpp"

namespace hyper {

struct PolicyQuery {
  // Definitions.
  using Weights = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  // Members.
  const InterpolatorLayout& layout;
  const Pointers<const Scalar>& inputs;
  const Weights& weights;
};

class AbstractPolicy;

template <typename>
class CartesianPolicy;

template <typename>
class ManifoldPolicy;

} // namespace hyper
