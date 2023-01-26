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

  using Scalar = TScalar;
  using Input = variables::SE3<TScalar>;
  using Output = variables::SE3<TScalar>;

  /// Evaluates this.
  /// \param inputs Inputs.
  /// \param weights Weights.
  /// \param jacobians Jacobian flag.
  /// \param input_offset Input offset.
  /// \param num_input_parameters Number of input parameters.
  /// \return Result.
  static auto evaluate(const std::vector<const Scalar*>& inputs, const Eigen::Ref<const MatrixX<Scalar>>& weights, bool jacobians, const Index& input_offset = 0,
                       const Index& num_input_parameters = Input::kNumParameters) -> Result<Output>;
};

}  // namespace hyper::state
