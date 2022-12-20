/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/motion/interpolators/spatial/forward.hpp"
#include "hyper/variables/groups/se3.hpp"

namespace hyper {

template <>
class SpatialInterpolator<Stamped<SE3<Scalar>>> final {
 public:
  // Definitions.
  using Input = Stamped<SE3<Scalar>>;

  // Definitions.
  using Index = Eigen::Index;
  // using Scalar = typename SE3<Scalar>::Scalar;

  using Manifold = SE3<Scalar>;
  using Tangent = hyper::Tangent<SE3<Scalar>>;

  // Constants.
  static constexpr auto kDimManifold = Manifold::kNumParameters;
  static constexpr auto kDimTangent = Tangent::kNumParameters;

  /// Evaluates this.
  /// \param query Spatial interpolator query.
  /// \return True on success.
  [[nodiscard]] static auto evaluate(const SpatialInterpolatorQuery& query) -> bool;

 private:
  template <MotionDerivative TMotionDerivative>
  [[nodiscard]] static auto evaluate(const SpatialInterpolatorQuery& query) -> bool;
};

} // namespace hyper
