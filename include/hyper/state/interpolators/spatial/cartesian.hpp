/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/forward.hpp"
#include "hyper/state/interpolators/forward.hpp"

#include "hyper/variables/cartesian.hpp"
#include "hyper/variables/jacobian.hpp"

namespace hyper::state {

template <typename TVariable>
class SpatialInterpolator<TVariable> final {
 public:
  // Definitions.
  using Index = Eigen::Index;

  using Input = TVariable;
  using Output = TVariable;

  using Scalar = typename TVariable::Scalar;
  using Inputs = std::vector<const Scalar*>;
  using Weights = Eigen::Ref<const MatrixX<Scalar>>;

  /// Evaluates this.
  /// \param inputs Inputs.
  /// \param weights Weights.
  /// \param jacobians Jacobian flag.
  /// \param offset Offset.
  /// \param stride Stride (i.e. number of input parameters).
  /// \return Result.
  static auto evaluate(const Inputs& inputs, const Weights& weights, bool jacobians, const Index& offset = 0, const Index& stride = Input::kNumParameters) -> Result<Output> {
    // Constants.
    const auto num_variables = weights.rows();
    const auto num_derivatives = weights.cols();

    const auto degree = num_derivatives - 1;
    const auto idx = offset + num_variables - 1;

    // Allocate result.
    auto result = Result<Output>(degree, jacobians, inputs.size(), stride);

    // Compute increments.
    constexpr auto kNumTangentParameters = variables::Tangent<Output>::kNumParameters;
    auto increments = Eigen::Matrix<Scalar, kNumTangentParameters, Eigen::Dynamic>{kNumTangentParameters, num_variables};
    increments.col(0).noalias() = Eigen::Map<const Input>{inputs[offset]};

    for (auto i = offset; i < idx; ++i) {
      increments.col(i - offset + 1).noalias() = Eigen::Map<const Input>{inputs[i + 1]} - Eigen::Map<const Input>{inputs[i]};
    }

    for (Index k = 0; k < num_derivatives; ++k) {
      // Evaluate values.
      if (k == 0) {
        result.value() = increments * weights.col(0);
      } else {
        result.derivative(k - 1) = increments * weights.col(k);
      }

      // Evaluate Jacobians.
      if (jacobians) {
        if (inputs.size() > 1) {
          if (k == 0) {
            result.jacobian(0, offset).diagonal().setConstant(Scalar{1} - weights(1, k));
          } else {
            result.jacobian(k, offset).diagonal().setConstant(Scalar{-1} * weights(1, k));
          }
          for (auto i = offset + 1; i < idx; ++i) {
            result.jacobian(k, i).diagonal().setConstant(weights(i, k) - weights(i + 1, k));
          }
          result.jacobian(k, idx).diagonal().setConstant(weights(idx, k));
        } else if (k == 0) {
          result.jacobian(0, offset).diagonal().setConstant(Scalar{1});
        }
      }
    }

    return result;
  }
};

}  // namespace hyper::state
