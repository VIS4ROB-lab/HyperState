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

  /// Evaluates this.
  /// \param query Spatial interpolator query.
  /// \return True on success.
  [[nodiscard]] static auto evaluate(const SpatialInterpolatorQuery& query) -> StateResult;

 private:
  template <int TDerivative>
  [[nodiscard]] static auto evaluate(const SpatialInterpolatorQuery& query) -> StateResult;
};

} // namespace hyper
