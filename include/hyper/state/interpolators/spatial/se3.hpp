/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/interpolators/forward.hpp"

#include "hyper/variables/groups/se3.hpp"

namespace hyper::state {

template <typename TScalar>
class SpatialInterpolator<variables::SE3<TScalar>> final {
 public:
  // Definitions.
  using Index = Eigen::Index;

  using Input = variables::SE3<TScalar>;
  using Output = variables::SE3<TScalar>;

  using Scalar = TScalar;
  using Inputs = std::vector<const TScalar*>;
  using Weights = Eigen::Ref<const MatrixX<TScalar>>;

  /// Evaluates this.
  /// \param inputs Inputs.
  /// \param weights Weights.
  /// \param jacobians Jacobian flag.
  /// \param offset Offset.
  /// \param stride Stride (i.e. number of input parameters).
  /// \return Result.
  static auto evaluate(const Inputs& inputs, const Weights& weights, bool jacobians, const Index& offset = 0, const Index& stride = Input::kNumParameters) -> Result<Output>;
};

}  // namespace hyper::state
