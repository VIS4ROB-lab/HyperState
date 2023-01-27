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
  static auto evaluate(const Index& derivative, const Scalar* const* inputs, const Index& num_inputs, const Index& start_index, const Index& end_index,
                       const Index& num_input_parameters, const Index& input_offset, const Eigen::Ref<const MatrixX<Scalar>>& weights, bool jacobians) -> Result<Output> {
    // Constants.
    constexpr auto kValue = 0;
    //constexpr auto kVelocity = 1;
    //constexpr auto kAcceleration = 2;

    // Allocate result.
    auto result = Result<Output>(derivative, jacobians, num_inputs, num_input_parameters);

    // Input lambda definition.
    auto I = [&inputs, &input_offset](const Index& i) {
      return Eigen::Map<const Input>{inputs[i] + input_offset};
    };

    // Compute increments.
    auto values = MatrixNX<Output>{Output::kNumParameters, weights.rows()};
    values.col(0).noalias() = I(start_index);
    for (auto i = start_index; i < end_index; ++i) {
      values.col(i - start_index + 1).noalias() = I(i + 1) - I(i);
    }

    // Compute value and derivatives.
    if (derivative == kValue) {
      result.value = values * weights.col(kValue);
    } else {
      result.value = values * weights.col(kValue);
      result.derivatives() = values * weights.rightCols(derivative);
    }

    // Compute Jacobians.
    if (jacobians) {
      // Jacobian lambda definition.
      auto J = [&result, &input_offset](const Index& k, const Index& i) {
        return result.template jacobian<Output::kNumParameters, Input::kNumParameters>(k, i, 0, input_offset);
      };

      for (Index k = kValue; k <= derivative; ++k) {
        for (auto i = start_index; i < end_index; ++i) {
          J(k, i).diagonal().array() = weights(i, k) - weights(i + 1, k);
        }
        J(k, end_index).diagonal().array() = weights(end_index, k);
      }
    }

    return result;
  }
};

}  // namespace hyper::state
