/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <iostream>

#include <glog/logging.h>

#include "hyper/matrix.hpp"
#include "hyper/state/continuous.hpp"
#include "hyper/variables/se3.hpp"
#include "hyper/variables/stamped.hpp"

namespace hyper::state {

using namespace variables;

template <typename TGroup>
ContinuousState<TGroup>::ContinuousState(std::unique_ptr<TemporalInterpolator>&& interpolator, bool uniform, const Jacobian jacobian)
    : TemporalState<TGroup>{uniform, jacobian}, layout_{}, interpolator_{} {
  swapInterpolator(interpolator);
}

template <typename TGroup>
auto ContinuousState<TGroup>::setUniform(bool flag) -> void {
  TemporalState<TGroup>::setUniform(flag);
  layout_ = interpolator_->layout(flag);
}

template <typename TGroup>
auto ContinuousState<TGroup>::range() const -> InclusiveRange<Scalar> {
  DCHECK_LE(layout_.outer_size, this->stamped_parameters_.size());
  const auto v0_itr = std::next(this->stamped_parameters_.cbegin(), layout_.left_margin - 1);
  const auto vn_itr = std::next(this->stamped_parameters_.crbegin(), layout_.right_margin - 1);
  const auto t0 = v0_itr->time();
  const auto tn = vn_itr->time();
  DCHECK_LT(t0, tn);
  return {t0, std::nexttoward(tn, t0)};
}

template <typename TGroup>
auto ContinuousState<TGroup>::partition(const Time& time) const -> Partition<Scalar*> {
  const auto [begin, end, num_stamped_parameters] = iterators(time);
  Partition<Scalar*> partition;
  partition.reserve(num_stamped_parameters);
  const auto input_size = this->tangentInputSize();
  std::transform(begin, end, std::back_inserter(partition), [&input_size](const auto& stamped_parameter) -> Partition<Scalar*>::value_type {
    return {const_cast<Scalar*>(stamped_parameter.data()), input_size};
  });
  return partition;
}

template <typename TGroup>
auto ContinuousState<TGroup>::parameterBlocks(const Time& time) const -> std::vector<Scalar*> {
  const auto [begin, end, num_variables] = iterators(time);
  std::vector<Scalar*> ptrs;
  ptrs.reserve(num_variables);
  std::transform(begin, end, std::back_inserter(ptrs), [](const auto& parameter) { return const_cast<Scalar*>(parameter.data()); });
  return ptrs;
}

template <typename TGroup>
auto ContinuousState<TGroup>::interpolator() const -> const TemporalInterpolator& {
  return *interpolator_;
}

template <typename TGroup>
auto ContinuousState<TGroup>::swapInterpolator(std::unique_ptr<TemporalInterpolator>& interpolator) -> void {
  CHECK(interpolator != nullptr);
  layout_ = interpolator->layout(this->uniform_);
  interpolator_.swap(interpolator);
}

template <typename TGroup>
auto ContinuousState<TGroup>::layout() const -> const TemporalInterpolatorLayout& {
  return layout_;
}

template <typename TGroup>
auto ContinuousState<TGroup>::evaluate(const Time& time, int derivative, bool jacobian, const Scalar* const* stamped_parameters) const -> Result<TGroup> {
  if (!stamped_parameters) {
    const auto [begin, end, num_parameters] = iterators(time);
    std::vector<const Scalar*> ptrs;
    ptrs.reserve(num_parameters);
    std::transform(begin, end, std::back_inserter(ptrs), [](const auto& parameter) { return parameter.data(); });
    DCHECK_EQ(ptrs.size(), num_parameters);
    return evaluate(time, derivative, jacobian, ptrs.data());

  } else {
    // Constants.
    constexpr auto kStampOffset = Stamped<TGroup>::kStampOffset;
    constexpr auto kVariableOffset = Stamped<TGroup>::kVariableOffset;

    // Fetch layout.
    const auto s_idx = layout_.left_padding;
    const auto e_idx = layout_.left_padding + layout_.inner_size - 1;

    // Compute normalized time.
    const auto idx = layout_.left_margin - 1;
    const auto dt = time - stamped_parameters[idx][kStampOffset];
    const auto i_dt = Scalar{1} / (stamped_parameters[idx + 1][kStampOffset] - stamped_parameters[idx][kStampOffset]);
    const auto ut = dt * i_dt;

    // Evaluate output.
    auto result = Result<TGroup>{derivative, layout_.outer_size, this->tangentInputSize(), jacobian};
    const auto weights = interpolator_->evaluate(ut, i_dt, derivative, !this->uniform_ ? stamped_parameters : nullptr, kStampOffset);
    SpatialInterpolator<TGroup>::evaluate(result, weights.data(), stamped_parameters, s_idx, e_idx, kVariableOffset);

    // Convert Jacobians.
    if (jacobian && (this->jacobian_ == Jacobian::TANGENT_TO_GROUP || this->jacobian_ == Jacobian::TANGENT_TO_STAMPED_GROUP)) {
      for (auto i = s_idx; i <= e_idx; ++i) {
        const auto J_a = Eigen::Map<const TGroup>{stamped_parameters[i] + kVariableOffset}.tMinusJacobian();
        for (auto k = 0; k <= derivative; ++k) {
          result.template jacobian<Tangent<TGroup>::kNumParameters, TGroup::kNumParameters>(k, i, 0, kVariableOffset) =
              result.template jacobian<Tangent<TGroup>::kNumParameters, Tangent<TGroup>::kNumParameters>(k, i, 0, kVariableOffset) * J_a;
        }
      }
    }

    return result;
  }
}

template <typename TGroup>
auto ContinuousState<TGroup>::iterators(const Time& time) const -> std::tuple<Iterator, Iterator, int> {
  DCHECK(range().contains(time)) << "Range does not contain time.";
  DCHECK_LE(layout_.outer_size, this->stamped_parameters_.size());
  const auto itr = this->stamped_parameters_.upper_bound(time);
  const auto begin = std::prev(itr, layout_.left_margin);
  const auto end = std::next(itr, layout_.right_margin);
  return {begin, end, layout_.outer_size};
}

template class ContinuousState<R3>;
template class ContinuousState<SU2>;
template class ContinuousState<SE3>;

}  // namespace hyper::state
