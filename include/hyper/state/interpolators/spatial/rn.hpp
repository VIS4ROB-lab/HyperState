/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/interpolators/forward.hpp"

namespace hyper::state {

template <typename TScalar, int N>
class SpatialInterpolator<variables::Rn<TScalar, N>> final {
 public:
  // Definitions.
  using Scalar = TScalar;
  using Input = variables::Rn<TScalar, N>;
  using Output = variables::Rn<TScalar, N>;

  /// Evaluates this.
  static auto evaluate(Result<Output>& result, const Scalar* weights, const Scalar* const* inputs, int s_idx, int e_idx, int offs) -> void;
};

}  // namespace hyper::state
