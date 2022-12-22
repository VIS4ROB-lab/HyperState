/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/motion/interpolators/spatial/forward.hpp"
#include "hyper/motion/interpolators/temporal/forward.hpp"

#include "hyper/variables/groups/se3.hpp"

namespace hyper {

template <>
class SpatialInterpolator<Stamped<SE3<Scalar>>> final {
 public:
  // Definitions.
  using Index = Eigen::Index;
  // using Scalar = typename SE3<Scalar>::Scalar;

  using Manifold = SE3<Scalar>;
  using Tangent = hyper::Tangent<SE3<Scalar>>;

  using Variables = Pointers<const Scalar>;
  using Weights = Eigen::Ref<const MatrixX<Scalar>>;
  using Outputs = Pointers<Scalar>;
  using Jacobians = std::vector<Pointers<Scalar>>;

  // Constants.
  static constexpr auto kDimManifold = Manifold::kNumParameters;
  static constexpr auto kDimTangent = Tangent::kNumParameters;

  /// Evaluates this.
  /// \param weights Interpolation weights.
  /// \param variables Interpolation variables.
  /// \param offset Offset into variables.
  /// \param jacobians Jacobians evaluation flag.
  /// \return Temporal motion results.
  static auto evaluate(const Weights& weights, const Variables& variables, const Outputs& outputs, const Jacobians* jacobians, const Index& offset) -> bool;

 private:
  template <MotionDerivative TMotionDerivative>
  static auto evaluate(const Weights& weights, const Variables& variables, const Outputs& outputs, const Jacobians* jacobians, const Index& offset) -> bool;
};

} // namespace hyper
