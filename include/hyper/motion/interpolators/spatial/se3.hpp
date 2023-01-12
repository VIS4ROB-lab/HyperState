/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/motion/interpolators/spatial/forward.hpp"
#include "hyper/motion/interpolators/temporal/forward.hpp"

#include "hyper/variables/groups/se3.hpp"

namespace hyper {

template <typename TScalar>
class SpatialInterpolator<SE3<TScalar>> final {
 public:
  // Definitions.
  using Index = Eigen::Index;
  using Scalar = TScalar;

  using Manifold = SE3<TScalar>;
  using Tangent = hyper::Tangent<SE3<TScalar>>;
  using Weights = Eigen::Ref<const MatrixX<TScalar>>;

  // Constants.
  static constexpr auto kDimManifold = Manifold::kNumParameters;
  static constexpr auto kDimTangent = Tangent::kNumParameters;

  /// Evaluate this.
  /// \param weights Weights.
  /// \param variables Variables.
  /// \param outputs Outputs.
  /// \param jacobians Jacobians.
  /// \param offset Offset.
  /// \param stride Jacobian stride.
  /// \return True on success.
  static auto evaluate(const Weights& weights, const Pointers<const TScalar>& variables, const Pointers<TScalar>& outputs, const Pointers<TScalar>* jacobians, const Index& offset, const Index& stride = kDimManifold) -> bool;
};

} // namespace hyper
