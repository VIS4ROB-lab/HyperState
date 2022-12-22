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

  using Variables = Pointers<const TScalar>;
  using Weights = Eigen::Ref<const MatrixX<TScalar>>;
  using Outputs = Pointers<TScalar>;
  using Jacobians = std::vector<Pointers<TScalar>>;

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
};

} // namespace hyper
