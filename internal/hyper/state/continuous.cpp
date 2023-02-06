/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <iostream>

#include <glog/logging.h>

#include "hyper/matrix.hpp"
#include "hyper/state/continuous.hpp"
#include "hyper/variables/groups/adapters.hpp"
#include "hyper/variables/groups/se3.hpp"
#include "hyper/variables/stamped.hpp"

namespace hyper::state {

using namespace variables;

template <typename TOutput, typename TVariable>
ContinuousState<TOutput, TVariable>::ContinuousState() = default;

template <typename TOutput, typename TVariable>
ContinuousState<TOutput, TVariable>::ContinuousState(const TemporalInterpolator<Scalar>* interpolator) {
  setInterpolator(interpolator);
}

template <typename TOutput, typename TVariable>
auto ContinuousState<TOutput, TVariable>::range() const -> Range {
  const auto layout = interpolator()->layout(this->isUniform());
  DCHECK_LE(layout.outer_input_size, this->stamped_variables_.size());
  const auto v0_itr = std::next(this->stamped_variables_.cbegin(), layout.left_input_margin - 1);
  const auto vn_itr = std::next(this->stamped_variables_.crbegin(), layout.right_input_margin - 1);
  DCHECK_LT(v0_itr->time(), vn_itr->time());
  return {v0_itr->time(), vn_itr->time()};
}

template <typename TOutput, typename TVariable>
auto ContinuousState<TOutput, TVariable>::variables() const -> std::vector<StampedVariable*> {
  std::vector<StampedVariable*> ptrs;
  ptrs.reserve(this->stamped_variables_.size());
  std::transform(this->stamped_variables_.begin(), this->stamped_variables_.end(), std::back_inserter(ptrs),
                 [](const auto& element) { return const_cast<StampedVariable*>(&element); });
  return ptrs;
}

template <typename TOutput, typename TVariable>
auto ContinuousState<TOutput, TVariable>::variables(const Time& time) const -> std::vector<StampedVariable*> {
  const auto& [begin, end, num_variables] = iterators(time);
  std::vector<StampedVariable*> ptrs;
  ptrs.reserve(num_variables);
  std::transform(begin, end, std::back_inserter(ptrs), [](const auto& element) { return const_cast<StampedVariable*>(&element); });
  return ptrs;
}

template <typename TOutput, typename TVariable>
auto ContinuousState<TOutput, TVariable>::parameterBlocks() const -> std::vector<Scalar*> {
  std::vector<Scalar*> ptrs;
  ptrs.reserve(this->stamped_variables_.size());
  std::transform(this->stamped_variables_.begin(), this->stamped_variables_.end(), std::back_inserter(ptrs),
                 [](const auto& element) { return const_cast<Scalar*>(element.data()); });
  return ptrs;
}

template <typename TOutput, typename TVariable>
auto ContinuousState<TOutput, TVariable>::parameterBlocks(const Time& time) const -> std::vector<Scalar*> {
  const auto& [begin, end, num_variables] = iterators(time);
  std::vector<Scalar*> ptrs;
  ptrs.reserve(num_variables);
  std::transform(begin, end, std::back_inserter(ptrs), [](const auto& element) { return const_cast<Scalar*>(element.data()); });
  return ptrs;
}

template <typename TOutput, typename TVariable>
auto ContinuousState<TOutput, TVariable>::interpolator() const -> const TemporalInterpolator<Scalar>* {
  return interpolator_;
}

template <typename TOutput, typename TVariable>
auto ContinuousState<TOutput, TVariable>::setInterpolator(const TemporalInterpolator<Scalar>* interpolator) -> void {
  DCHECK(interpolator != nullptr);
  interpolator_ = interpolator;
}

template <typename TOutput, typename TVariable>
auto ContinuousState<TOutput, TVariable>::evaluate(const Time& time, const Index& derivative, JacobianType jacobian_type,  // NOLINT
                                                   const Scalar* const* stamped_variables) const -> Result<TOutput> {
  if (!stamped_variables) {
    const auto& [begin, end, num_variables] = iterators(time);
    std::vector<const Scalar*> ptrs;
    ptrs.reserve(num_variables);
    std::transform(begin, end, std::back_inserter(ptrs), [](const auto& element) { return element.data(); });
    DCHECK_EQ(ptrs.size(), num_variables);
    return evaluate(time, derivative, jacobian_type, ptrs.data());
  } else {
    // Constants.
    constexpr auto kStampOffset = StampedVariable::kStampOffset;
    constexpr auto kVariableOffset = StampedVariable::kVariableOffset;

    // Fetch layout.
    const auto layout = interpolator()->layout(this->isUniform());
    const auto s_idx = layout.left_input_padding;
    const auto e_idx = layout.left_input_padding + layout.inner_input_size - 1;

    // Compute normalized time.
    const auto idx = layout.left_input_margin - 1;
    const auto dt = time - stamped_variables[idx][kStampOffset];
    const auto i_dt = Scalar{1} / (stamped_variables[idx + 1][kStampOffset] - stamped_variables[idx][kStampOffset]);
    const auto ut = dt * i_dt;

    if (this->isUniform()) {
      // Evaluate uniform weights.
      const auto weights = interpolator()->evaluate(ut, i_dt, derivative, nullptr, kStampOffset);

      if (jacobian_type != JacobianType::TANGENT_TO_PARAMETERS) {
        // Evaluation with tangent to tangent Jacobians.
        auto result = Result<Output>{derivative, jacobian_type, layout.outer_input_size, OutputTangent::kNumParameters};
        SpatialInterpolator<TOutput, TVariable>::evaluate(result, weights, stamped_variables, s_idx, e_idx, kVariableOffset);
        return result;

      } else {
        // Evaluation with tangent to group Jacobians.
        auto result = Result<Output>{derivative, jacobian_type, layout.outer_input_size, Variable::kNumParameters};
        SpatialInterpolator<TOutput, TVariable>::evaluate(result, weights, stamped_variables, s_idx, e_idx, kVariableOffset);

        // Lift tangent to group/variable Jacobians.
        for (auto i = s_idx; i <= e_idx; ++i) {
          const auto J_a = JacobianAdapter<Variable>(stamped_variables[i] + kVariableOffset);
          for (auto k = 0; k <= derivative; ++k) {
            result.jacobian(k, i) = result.template jacobian<OutputTangent::kNumParameters, VariableTangent::kNumParameters>(k, i, 0, kVariableOffset) * J_a;
          }
        }

        return result;
      }
    } else {
      // Evaluate non-uniform weights.
      const auto weights = interpolator()->evaluate(ut, i_dt, derivative, stamped_variables, kStampOffset);

      if (jacobian_type != JacobianType::TANGENT_TO_PARAMETERS) {
        // Evaluation with tangent to tangent Jacobians.
        auto result = Result<Output>{derivative, jacobian_type, layout.outer_input_size, StampedOutputTangent::kNumParameters};
        SpatialInterpolator<TOutput, TVariable>::evaluate(result, weights, stamped_variables, s_idx, e_idx, kVariableOffset);
        return result;

      } else {
        // Evaluation with tangent to group/variable Jacobians.
        auto result = Result<Output>{derivative, jacobian_type, layout.outer_input_size, StampedVariable::kNumParameters};
        SpatialInterpolator<TOutput, TVariable>::evaluate(result, weights, stamped_variables, s_idx, e_idx, kVariableOffset);

        // Lift tangent to group Jacobians.
        for (auto i = s_idx; i <= e_idx; ++i) {
          const auto J_a = JacobianAdapter<StampedVariable>(stamped_variables[i]);
          for (auto k = 0; k <= derivative; ++k) {
            result.jacobian(k, i) = result.template jacobian<OutputTangent::kNumParameters, StampedVariableTangent::kNumParameters>(k, i, 0, 0) * J_a;
          }
        }

        return result;
      }
    }
  }
}

template <typename TOutput, typename TVariable>
auto ContinuousState<TOutput, TVariable>::iterators(const Time& time) const -> std::tuple<Iterator, Iterator, Index> {
  DCHECK(range().contains(time)) << "Range does not contain time.";
  const auto layout = interpolator()->layout(this->isUniform());

  DCHECK_LE(layout.outer_input_size, this->stamped_variables_.size());
  const auto itr = this->stamped_variables_.upper_bound(time);
  const auto begin = std::prev(itr, layout.left_input_margin);
  const auto end = std::next(itr, layout.right_input_margin);
  return {begin, end, layout.outer_input_size};
}

template class ContinuousState<Cartesian<double, 3>>;
template class ContinuousState<SU2<double>>;
template class ContinuousState<SE3<double>>;

}  // namespace hyper::state
