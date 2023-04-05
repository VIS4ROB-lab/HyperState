/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/interpolators/forward.hpp"
#include "hyper/variables/forward.hpp"

namespace hyper::state {

template <typename TScalar>
class SpatialInterpolator<variables::SE3<TScalar>> final {
 public:
  // Definitions.
  using Scalar = TScalar;
  using Input = variables::SE3<TScalar>;
  using Output = variables::SE3<TScalar>;

  /// Evaluates this.
  static auto evaluate(Result<Output>& result, const TScalar* weights, const TScalar* const* inputs, int s_idx, int e_idx, int offs) -> void;
};

template <typename TScalar>
using SE3Interpolator = SpatialInterpolator<variables::SE3<TScalar>>;

}  // namespace hyper::state
