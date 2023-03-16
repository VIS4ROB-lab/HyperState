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
ContinuousState<TOutput, TVariable>::ContinuousState(std::unique_ptr<TemporalInterpolator<Scalar>>&& interpolator, bool is_uniform, const JacobianType jacobian_type)
    : Base{is_uniform, jacobian_type}, layout_{}, interpolator_{} {
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
auto ContinuousState<TOutput, TVariable>::evaluate(const Time& time, int derivative, bool jacobian, const Scalar* const* stamped_variables) const -> Result<TOutput> {
  if (!stamped_variables) {
    const auto& [begin, end, num_variables] = iterators(time);
    std::vector<const Scalar*> ptrs;
    ptrs.reserve(num_variables);
    std::transform(begin, end, std::back_inserter(ptrs), [](const auto& element) { return element.data(); });
    DCHECK_EQ(ptrs.size(), num_variables);
    return evaluate(time, derivative, jacobian, ptrs.data());

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

    // Evaluate output.
    auto result = Result<Output>{derivative, layout_.outer_size, this->localInputSize(), jacobian};
    const auto weights = interpolator_->evaluate(ut, i_dt, derivative, !this->is_uniform_ ? stamped_variables : nullptr, kStampOffset);
    SpatialInterpolator<TOutput, TVariable>::evaluate(result, weights.data(), stamped_variables, s_idx, e_idx, kVariableOffset);

    // Convert Jacobians.
    if (jacobian && (this->jacobian_type_ == JacobianType::TANGENT_TO_MANIFOLD || this->jacobian_type_ == JacobianType::TANGENT_TO_STAMPED_MANIFOLD)) {
      for (auto i = s_idx; i <= e_idx; ++i) {
        const auto J_a = JacobianAdapter<Variable>(stamped_variables[i] + kVariableOffset);
        for (auto k = 0; k <= derivative; ++k) {
          result.template jacobian<OutputTangent::kNumParameters, Variable::kNumParameters>(k, i, 0, kVariableOffset) =
              result.template jacobian<OutputTangent::kNumParameters, VariableTangent::kNumParameters>(k, i, 0, kVariableOffset) * J_a;
        }
      }
    }

    return result;
  }
}

template <typename TOutput, typename TVariable>
auto ContinuousState<TOutput, TVariable>::iterators(const Time& time) const -> std::tuple<Iterator, Iterator, int> {
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
