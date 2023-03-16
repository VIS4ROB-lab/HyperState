/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/interpolators/forward.hpp"
#include "hyper/variables/forward.hpp"

namespace hyper::state {

template <typename TScalar, int TSize>
class SpatialInterpolator<variables::Cartesian<TScalar, TSize>> final {
 public:
  // Definitions.
  using Scalar = TScalar;
  using Input = variables::Cartesian<TScalar, TSize>;
  using Output = variables::Cartesian<TScalar, TSize>;

  /// Evaluates this.
  static auto evaluate(Result<Output>& result, const Scalar* weights, const Scalar* const* inputs, int s_idx, int e_idx, int offs) -> void;
};

template <typename TScalar, int TSize>
using CartesianInterpolator = SpatialInterpolator<variables::Cartesian<TScalar, TSize>>;

}  // namespace hyper::state
