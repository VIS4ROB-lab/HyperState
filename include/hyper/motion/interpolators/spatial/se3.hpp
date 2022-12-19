/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/motion/forward.hpp"
#include "hyper/motion/interpolators/spatial/forward.hpp"
#include "hyper/variables/groups/forward.hpp"

namespace hyper {

template <>
class SpatialInterpolator<Stamped<SE3<Scalar>>> final {
 public:
  // Definitions.
  using Value = SE3<Scalar>;
  using Input = Stamped<Value>;
  using Derivative = Tangent<Value>;

  /// Evaluates a query.
  /// \param state_query State query.
  /// \param policy_query Policy query.
  /// \return Interpolation result.
  [[nodiscard]] static auto evaluate(const StateQuery& state_query, const PolicyQuery& policy_query) -> StateResult;
};

} // namespace hyper
