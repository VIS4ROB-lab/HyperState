/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/interpolators/forward.hpp"
#include "hyper/variables/groups/forward.hpp"

namespace hyper::state {

template <typename TScalar>
class SpatialInterpolator<variables::SU2<TScalar>> final {
 public:
  // Definitions.
  using Scalar = TScalar;
  using Input = variables::SU2<TScalar>;
  using Output = variables::SU2<TScalar>;

  /// Evaluates this.
  static auto evaluate(Result<Output>& result, const Scalar* weights, const TScalar* const* inputs, int s_idx, int e_idx, int offs) -> void;
};

template <typename TScalar>
using SU2Interpolator = SpatialInterpolator<variables::SU2<TScalar>>;

}  // namespace hyper::state
