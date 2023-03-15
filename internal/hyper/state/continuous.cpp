/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <iostream>

#include <glog/logging.h>

#include "hyper/matrix.hpp"
#include "hyper/state/continuous.hpp"
#include "hyper/variables/adapters.hpp"
#include "hyper/variables/groups/se3.hpp"
#include "hyper/variables/stamped.hpp"

namespace hyper::state {

using namespace variables;

template <typename TOutput, typename TVariable>
ContinuousState<TOutput, TVariable>::ContinuousState(std::unique_ptr<TemporalInterpolator<Scalar>>&& interpolator) {
  swapInterpolator(interpolator);
}

template <typename TOutput, typename TVariable>
auto ContinuousState<TOutput, TVariable>::setUniform(bool flag) -> void {
  Base::setUniform(flag);
  layout_ = interpolator_->layout(flag);
}

template <typename TOutput, typename TVariable>
auto ContinuousState<TOutput, TVariable>::range() const -> Range {
  DCHECK_LE(layout_.outer_size, this->stamped_variables_.size());
  const auto v0_itr = std::next(this->stamped_variables_.cbegin(), layout_.left_margin - 1);
  const auto vn_itr = std::next(this->stamped_variables_.crbegin(), layout_.right_margin - 1);
  const auto t0 = v0_itr->time();
  const auto tn = vn_itr->time();
  DCHECK_LT(t0, tn);
  return {t0, std::nexttoward(tn, t0)};
}

template <typename TOutput, typename TVariable>
auto ContinuousState<TOutput, TVariable>::variables() const -> std::vector<variables::Variable<Scalar>*> {
  std::vector<variables::Variable<Scalar>*> ptrs;
  ptrs.reserve(this->stamped_variables_.size());
  std::transform(this->stamped_variables_.begin(), this->stamped_variables_.end(), std::back_inserter(ptrs),
                 [](const auto& element) { return const_cast<StampedVariable*>(&element); });
  return ptrs;
}

template <typename TOutput, typename TVariable>
auto ContinuousState<TOutput, TVariable>::variables(const Time& time) const -> std::vector<variables::Variable<Scalar>*> {
  const auto& [begin, end, num_variables] = iterators(time);
  std::vector<variables::Variable<Scalar>*> ptrs;
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
auto ContinuousState<TOutput, TVariable>::interpolator() const -> const TemporalInterpolator<Scalar>& {
  return *interpolator_;
}

template <typename TOutput, typename TVariable>
auto ContinuousState<TOutput, TVariable>::swapInterpolator(std::unique_ptr<TemporalInterpolator<Scalar>>& interpolator) -> void {
  CHECK(interpolator != nullptr);
  layout_ = interpolator->layout(this->is_uniform_);
  interpolator_.swap(interpolator);
}

template <typename TOutput, typename TVariable>
auto ContinuousState<TOutput, TVariable>::layout() const -> const TemporalInterpolatorLayout& {
  return layout_;
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
    const auto s_idx = layout_.left_padding;
    const auto e_idx = layout_.left_padding + layout_.inner_size - 1;

    // Compute normalized time.
    const auto idx = layout_.left_margin - 1;
    const auto dt = time - stamped_variables[idx][kStampOffset];
    const auto i_dt = Scalar{1} / (stamped_variables[idx + 1][kStampOffset] - stamped_variables[idx][kStampOffset]);
    const auto ut = dt * i_dt;

    if (this->isUniform()) {
      // Evaluate uniform weights.
      const auto weights = interpolator_->evaluate(ut, i_dt, derivative, nullptr, kStampOffset);

      if (jacobian_type != JacobianType::TANGENT_TO_PARAMETERS) {
        // Evaluation with tangent to tangent Jacobians.
        auto result = Result<Output>{derivative, jacobian_type, layout_.outer_size, OutputTangent::kNumParameters};
        SpatialInterpolator<TOutput, TVariable>::evaluate(result, weights, stamped_variables, s_idx, e_idx, kVariableOffset);
        return result;

      } else {
        // Evaluation with tangent to parameter Jacobians.
        auto result = Result<Output>{derivative, jacobian_type, layout_.outer_size, Variable::kNumParameters};
        SpatialInterpolator<TOutput, TVariable>::evaluate(result, weights, stamped_variables, s_idx, e_idx, kVariableOffset);

        // Lift tangent to parameter Jacobians.
        for (auto i = s_idx; i <= e_idx; ++i) {
          const auto J_a = JacobianAdapter<Variable>(stamped_variables[i] + kVariableOffset);
          for (auto k = 0; k <= derivative; ++k) {
            result.template jacobian<OutputTangent::kNumParameters, Variable::kNumParameters>(k, i, 0, kVariableOffset) =
                result.template jacobian<OutputTangent::kNumParameters, VariableTangent::kNumParameters>(k, i, 0, kVariableOffset) * J_a;
          }
        }

        return result;
      }
    } else {
      // Evaluate non-uniform weights.
      const auto weights = interpolator_->evaluate(ut, i_dt, derivative, stamped_variables, kStampOffset);

      if (jacobian_type != JacobianType::TANGENT_TO_PARAMETERS) {
        // Evaluation with tangent to tangent Jacobians.
        auto result = Result<Output>{derivative, jacobian_type, layout_.outer_size, StampedOutputTangent::kNumParameters};
        SpatialInterpolator<TOutput, TVariable>::evaluate(result, weights, stamped_variables, s_idx, e_idx, kVariableOffset);
        return result;

      } else {
        // Evaluation with tangent to parameter Jacobians.
        auto result = Result<Output>{derivative, jacobian_type, layout_.outer_size, StampedVariable::kNumParameters};
        SpatialInterpolator<TOutput, TVariable>::evaluate(result, weights, stamped_variables, s_idx, e_idx, kVariableOffset);

        // Lift tangent to parameter Jacobians.
        for (auto i = s_idx; i <= e_idx; ++i) {
          const auto J_a = JacobianAdapter<Variable>(stamped_variables[i] + kVariableOffset);
          for (auto k = 0; k <= derivative; ++k) {
            result.template jacobian<OutputTangent::kNumParameters, Variable::kNumParameters>(k, i, 0, kVariableOffset) =
                result.template jacobian<OutputTangent::kNumParameters, VariableTangent::kNumParameters>(k, i, 0, kVariableOffset) * J_a;
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

  DCHECK_LE(layout_.outer_size, this->stamped_variables_.size());
  const auto itr = this->stamped_variables_.upper_bound(time);
  const auto begin = std::prev(itr, layout_.left_margin);
  const auto end = std::next(itr, layout_.right_margin);
  return {begin, end, layout_.outer_size};
}

template class ContinuousState<Cartesian<double, 3>>;
template class ContinuousState<SU2<double>>;
template class ContinuousState<SE3<double>>;

}  // namespace hyper::state
