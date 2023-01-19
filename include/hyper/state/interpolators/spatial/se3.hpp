/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/interpolators/spatial/forward.hpp"
#include "hyper/state/interpolators/temporal/forward.hpp"

#include "hyper/variables/groups/se3.hpp"

namespace hyper::state {

template <typename TScalar>
class SpatialInterpolator<variables::SE3<TScalar>> final {
 public:
  // Definitions.
  using Index = Eigen::Index;

  using Manifold = variables::SE3<TScalar>;
  using Tangent = variables::Tangent<Manifold>;
  using Jacobian = variables::JacobianNM<Tangent, Manifold>;

  using Inputs = std::vector<const TScalar*>;
  using Weights = Eigen::Ref<const MatrixX<TScalar>>;
  using Outputs = std::vector<TScalar*>;
  using Jacobians = std::vector<TScalar*>;

  // Constants.
  static constexpr auto kDimManifold = Manifold::kNumParameters;
  static constexpr auto kDimTangent = Tangent::kNumParameters;

  /// Evaluate this.
  /// \param inputs Inputs.
  /// \param weights Weights.
  /// \param outputs Outputs.
  /// \param jacobians Jacobians.
  /// \param offset Offset.
  /// \param stride Jacobian stride.
  /// \return True on success.
  static auto evaluate(const Inputs& inputs, const Weights& weights, const Outputs& outputs, const Jacobians* jacobians, const Index& offset, const Index& stride = kDimManifold)
      -> bool;
};

}  // namespace hyper::state
