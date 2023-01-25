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
  using ValueOutput = TVariable;
  using TangentOutput = TVariable;
  using Jacobian = variables::JacobianNM<TangentOutput, Input>;

  using Scalar = typename TVariable::Scalar;
  using Inputs = std::vector<const Scalar*>;
  using Weights = Eigen::Ref<const MatrixX<Scalar>>;
  using Outputs = std::vector<Scalar*>;
  using Jacobians = std::vector<Scalar*>;

  // Constants.
  static constexpr auto kDimInput = Input::kNumParameters;
  static constexpr auto kDimValueOutput = ValueOutput::kNumParameters;
  static constexpr auto kDimTangentOutput = TangentOutput::kNumParameters;

  /// Evaluate this.
  /// \param inputs Inputs.
  /// \param weights Weights.
  /// \param outputs Outputs.
  /// \param jacobians Jacobians.
  /// \param offset Offset.
  /// \param stride Jacobian stride.
  /// \return True on success.
  static auto evaluate(const Inputs& inputs, const Weights& weights, const Outputs& outputs, const Jacobians* jacobians, const Index& offset,
                       const Index& stride = Input::kNumParameters) -> bool {
    // Constants.
    const auto num_variables = weights.rows();
    const auto num_derivatives = weights.cols();
    const auto last_idx = offset + num_variables - 1;

    // Compute increments.
    auto increments = Eigen::Matrix<Scalar, kDimTangentOutput, Eigen::Dynamic>{kDimTangentOutput, num_variables};
    increments.col(0).noalias() = Eigen::Map<const Input>{inputs[offset]};

    for (auto i = offset; i < last_idx; ++i) {
      increments.col(i - offset + 1).noalias() = Eigen::Map<const Input>{inputs[i + 1]} - Eigen::Map<const Input>{inputs[i]};
    }

    for (Index k = 0; k < num_derivatives; ++k) {
      // Evaluate values.
      if (k == 0) {
        Eigen::Map<ValueOutput>{outputs[0]} = increments * weights.col(0);
      } else {
        Eigen::Map<TangentOutput>{outputs[k]} = increments * weights.col(k);
      }

      // Evaluate Jacobians.
      if (jacobians) {
        const auto increment = (kDimTangentOutput * stride);
        if (inputs.size() > 1) {
          if (k == 0) {
            Eigen::Map<Jacobian>{(*jacobians)[0] + offset * increment}.diagonal().setConstant(Scalar{1} - weights(1, k));
          } else {
            Eigen::Map<Jacobian>{(*jacobians)[k] + offset * increment}.diagonal().setConstant(Scalar{-1} * weights(1, k));
          }
          for (auto i = offset + 1; i < last_idx; ++i) {
            Eigen::Map<Jacobian>{(*jacobians)[k] + i * increment}.diagonal().setConstant(weights(i, k) - weights(i + 1, k));
          }
          Eigen::Map<Jacobian>{(*jacobians)[k] + last_idx * increment}.diagonal().setConstant(weights(last_idx, k));
        } else if (k == 0) {
          Eigen::Map<Jacobian>{(*jacobians)[0] + offset * increment}.diagonal().setConstant(Scalar{1});
        }
      }
    }

    return true;
  }
};

}  // namespace hyper::state
