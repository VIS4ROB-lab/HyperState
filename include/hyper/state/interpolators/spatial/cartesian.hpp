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

  using Scalar = typename TVariable::Scalar;
  using Input = TVariable;
  using Output = TVariable;

  /// Evaluates this.
  /// \param inputs Inputs.
  /// \param weights Weights.
  /// \param jacobians Jacobian flag.
  /// \param input_offset Input offset.
  /// \param num_input_parameters Number of input parameters.
  /// \return Result.
  static auto evaluate(const std::vector<const Scalar*>& inputs, const Eigen::Ref<const MatrixX<Scalar>>& weights, bool jacobians, const Index& input_offset = 0,
                       const Index& num_input_parameters = Input::kNumParameters) -> Result<Output> {
    // Constants.
    const auto num_variables = weights.rows();
    const auto num_derivatives = weights.cols();

    const auto degree = num_derivatives - 1;
    const auto last_index = input_offset + num_variables - 1;

    // Allocate result.
    auto result = Result<Output>(degree, jacobians, inputs.size(), num_input_parameters);

    // Compute increments.
    auto increments = MatrixNX<Output>{Output::kNumParameters, num_variables};
    increments.col(0).noalias() = Eigen::Map<const Input>{inputs[input_offset]};

    for (auto i = input_offset; i < last_index; ++i) {
      increments.col(i - input_offset + 1).noalias() = Eigen::Map<const Input>{inputs[i + 1]} - Eigen::Map<const Input>{inputs[i]};
    }

    for (Index k = 0; k < num_derivatives; ++k) {
      if (k == 0) {
        result.value() = increments * weights.col(0);
      } else {
        result.derivative(k - 1) = increments * weights.col(k);
      }

      if (jacobians) {
        // Definitions.
        auto J = [&result](const Index& k, const Index& i, const Index& input_offset = 0) {
          return result.template jacobian<Output::kNumParameters, Input::kNumParameters>(k, i, 0, input_offset);
        };

        if (inputs.size() > 1) {
          if (k == 0) {
            J(0, input_offset).diagonal().setConstant(Scalar{1} - weights(1, k));
          } else {
            J(k, input_offset).diagonal().setConstant(Scalar{-1} * weights(1, k));
          }
          for (auto i = input_offset + 1; i < last_index; ++i) {
            J(k, i).diagonal().setConstant(weights(i, k) - weights(i + 1, k));
          }
          J(k, last_index).diagonal().setConstant(weights(last_index, k));
        } else if (k == 0) {
          J(0, input_offset).diagonal().setConstant(Scalar{1});
        }
      }
    }

    return result;
  }
};

}  // namespace hyper::state
