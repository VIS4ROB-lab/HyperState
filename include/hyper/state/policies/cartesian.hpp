/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/forward.hpp"

#include "hyper/state/policies/abstract.hpp"

namespace hyper {

template <typename TVariable>
class CartesianPolicy<Stamped<TVariable>> final : public AbstractPolicy {
 public:
  // Definitions.
  using Value = TVariable;
  using Input = Stamped<Value>;
  using Derivative = Value;

  /// Collects the times.
  /// \return Times.
  [[nodiscard]] auto times(const Pointers<const Scalar>& pointers) const -> Times final {
    Times times;
    times.reserve(pointers.size());
    for (const auto& pointer : pointers) {
      const auto stamped = Eigen::Map<const Input>{pointer};
      times.emplace_back(stamped.stamp());
    }
    return times;
  }

  /// Evaluates a query.
  /// \param state_query State query.
  /// \param policy_query Policy query.
  /// \return Interpolation result.
  [[nodiscard]] auto evaluate(const StateQuery& state_query, const PolicyQuery& policy_query) const -> StateResult final {
    // Definitions.
    using Result = StateResult;

    // Unpack queries.
    const auto& [stamp, derivative, jacobian] = state_query;
    const auto& [layout, inputs, weights] = policy_query;

    // Sanity checks.
    const auto num_inputs = static_cast<Index>(inputs.size());
    DCHECK(num_inputs == layout.outer.size && weights.rows() == layout.inner.size && weights.cols() == derivative + 1);

    // Allocate result.
    Result result;
    auto& [outputs, jacobians] = result;
    outputs.reserve(derivative + 1);
    jacobians.reserve(derivative + 1);

    if (num_inputs == 1) {
      for (Index k = 0; k <= derivative; ++k) {
        if (k == 0) {
          outputs.emplace_back(Eigen::Map<const Input>{inputs[0]}.variable());
          if (jacobian) {
            jacobians.emplace_back(Result::Jacobian::Identity(kNumDerivativeParameters, kNumInputParameters));
          }
        } else {
          outputs.emplace_back(Result::Derivative::Zero(kNumValueParameters, 1));
          if (jacobian) {
            jacobians.emplace_back(Result::Jacobian::Zero(kNumDerivativeParameters, kNumInputParameters));
          }
        }
      }
    } else {
      // Compute indices.
      const auto start_idx = layout.inner.index;
      const auto end_idx = start_idx + layout.inner.size - 1;

      // Evaluate increments.
      using Increments = Eigen::Matrix<Scalar, kNumDerivativeParameters, Eigen::Dynamic>;
      auto increments = Increments{kNumDerivativeParameters, end_idx - start_idx + 1};
      increments.col(0).noalias() = Eigen::Map<const Input>{inputs[start_idx]}.variable();

      for (auto i = start_idx; i < end_idx; ++i) {
        const auto x = Eigen::Map<const Input>{inputs[i]};
        const auto y = Eigen::Map<const Input>{inputs[i + 1]};
        increments.col(i - start_idx + 1).noalias() = y.variable() - x.variable();
      }

      for (Index k = 0; k <= derivative; ++k) {
        outputs.emplace_back(increments * weights.col(k));

        if (jacobian) {
          jacobians.emplace_back(Result::Jacobian::Zero(kNumDerivativeParameters, num_inputs * kNumInputParameters));

          if (k == 0) {
            jacobians[k].template middleCols<kNumInputParameters>(start_idx * kNumInputParameters).diagonal().setConstant(Scalar{1} - weights(1, k));
          } else {
            jacobians[k].template middleCols<kNumInputParameters>(start_idx * kNumInputParameters).diagonal().setConstant(Scalar{-1} * weights(1, k));
          }

          for (auto j = start_idx + 1; j < end_idx; ++j) {
            jacobians[k].template middleCols<kNumInputParameters>(j * kNumInputParameters).diagonal().setConstant(weights(j, k) - weights(j + 1, k));
          }

          jacobians[k].template middleCols<kNumInputParameters>(end_idx * kNumInputParameters).diagonal().setConstant(weights(end_idx, k));
        }
      }
    }

    return result;
  }

 private:
  // Constants.
  static constexpr auto kNumValueParameters = Value::kNumParameters;
  static constexpr auto kNumInputParameters = Input::kNumParameters;
  static constexpr auto kNumDerivativeParameters = Derivative::kNumParameters;
};

} // namespace hyper
