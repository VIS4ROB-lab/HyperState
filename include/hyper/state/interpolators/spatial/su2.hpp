/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/interpolators/forward.hpp"

namespace hyper::state {

template <>
class SpatialInterpolator<variables::SU2> final {
 public:
  // Definitions.
  using Input = variables::SU2;
  using Output = variables::SU2;

  /// Evaluates this.
  static auto evaluate(Result<Output>& result, const Scalar* weights, const Scalar* const* inputs, int s_idx, int e_idx, int offs) -> void;
};

}  // namespace hyper::state
