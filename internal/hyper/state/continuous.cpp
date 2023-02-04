/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <iostream>

#include <glog/logging.h>

#include "hyper/matrix.hpp"
#include "hyper/state/continuous.hpp"
#include "hyper/variables/groups/se3.hpp"
#include "hyper/variables/stamped.hpp"

namespace hyper::state {

using namespace variables;

template <typename TOutput, typename TInput>
ContinuousState<TOutput, TInput>::ContinuousState() = default;

template <typename TOutput, typename TInput>
ContinuousState<TOutput, TInput>::ContinuousState(const TemporalInterpolator<Scalar>* interpolator) {
  setInterpolator(interpolator);
}

template <typename TOutput, typename TInput>
auto ContinuousState<TOutput, TInput>::range() const -> Range {
  const auto layout = interpolator()->layout(this->isUniform());
  DCHECK_LE(layout.outer_input_size, this->stamped_inputs_.size());
  const auto v0_itr = std::next(this->stamped_inputs_.cbegin(), layout.left_input_margin - 1);
  const auto vn_itr = std::next(this->stamped_inputs_.crbegin(), layout.right_input_margin - 1);
  DCHECK_LT(v0_itr->time(), vn_itr->time());
  return {v0_itr->time(), vn_itr->time()};
}

template <typename TOutput, typename TInput>
auto ContinuousState<TOutput, TInput>::inputs() const -> std::vector<StampedInput*> {
  std::vector<StampedInput*> ptrs;
  ptrs.reserve(this->stamped_inputs_.size());
  std::transform(this->stamped_inputs_.begin(), this->stamped_inputs_.end(), std::back_inserter(ptrs), [](const auto& element) { return const_cast<StampedInput*>(&element); });
  return ptrs;
}

template <typename TOutput, typename TInput>
auto ContinuousState<TOutput, TInput>::inputs(const Time& time) const -> std::vector<StampedInput*> {
  const auto& [begin, end, num_inputs] = iterators(time);
  std::vector<StampedInput*> ptrs;
  ptrs.reserve(num_inputs);
  std::transform(begin, end, std::back_inserter(ptrs), [](const auto& element) { return const_cast<StampedInput*>(&element); });
  return ptrs;
}

template <typename TOutput, typename TInput>
auto ContinuousState<TOutput, TInput>::parameterBlocks() const -> std::vector<Scalar*> {
  std::vector<Scalar*> ptrs;
  ptrs.reserve(this->stamped_inputs_.size());
  std::transform(this->stamped_inputs_.begin(), this->stamped_inputs_.end(), std::back_inserter(ptrs), [](const auto& element) { return const_cast<Scalar*>(element.data()); });
  return ptrs;
}

template <typename TOutput, typename TInput>
auto ContinuousState<TOutput, TInput>::parameterBlocks(const Time& time) const -> std::vector<Scalar*> {
  const auto& [begin, end, num_inputs] = iterators(time);
  std::vector<Scalar*> ptrs;
  ptrs.reserve(num_inputs);
  std::transform(begin, end, std::back_inserter(ptrs), [](const auto& element) { return const_cast<Scalar*>(element.data()); });
  return ptrs;
}

template <typename TOutput, typename TInput>
auto ContinuousState<TOutput, TInput>::interpolator() const -> const TemporalInterpolator<Scalar>* {
  return interpolator_;
}

template <typename TOutput, typename TInput>
auto ContinuousState<TOutput, TInput>::setInterpolator(const TemporalInterpolator<Scalar>* interpolator) -> void {
  DCHECK(interpolator != nullptr);
  interpolator_ = interpolator;
}

template <typename TOutput, typename TInput>
auto ContinuousState<TOutput, TInput>::evaluate(const Time& time, const Index& derivative, bool jacobians) const -> Result<TOutput> {
  const auto& [begin, end, num_inputs] = iterators(time);
  std::vector<const Scalar*> ptrs;
  ptrs.reserve(num_inputs);
  std::transform(begin, end, std::back_inserter(ptrs), [](const auto& element) { return element.data(); });
  DCHECK_EQ(ptrs.size(), num_inputs);
  return evaluate(time, derivative, jacobians, ptrs.data());
}

template <typename TOutput, typename TInput>
auto ContinuousState<TOutput, TInput>::evaluate(const Time& time, const Index& derivative, bool jacobians, const Scalar* const* inputs) const -> Result<TOutput> {
  // Constants.
  constexpr auto kStampOffset = StampedInput::kStampOffset;
  constexpr auto kVariableOffset = StampedInput::kVariableOffset;

  // Fetch layout.
  const auto layout = interpolator()->layout(this->isUniform());
  const auto s_idx = layout.left_input_padding;
  const auto e_idx = layout.left_input_padding + layout.inner_input_size - 1;

  // Compute normalized time.
  const auto offset = layout.left_input_margin - 1;
  const auto dt = time - inputs[offset][kStampOffset];
  const auto i_dt = Scalar{1} / (inputs[offset + 1][kStampOffset] - inputs[offset][kStampOffset]);
  const auto ut = dt * i_dt;

  if (this->isUniform()) {
    const auto weights = interpolator()->evaluate(ut, i_dt, derivative, nullptr, kStampOffset);
    auto result = Result<Output>{derivative, jacobians, layout.outer_input_size, OutputTangent::kNumParameters};
    SpatialInterpolator<TOutput, TInput>::evaluate(result, weights, inputs, s_idx, e_idx, kVariableOffset);
    return result;
  } else {
    const auto weights = interpolator()->evaluate(ut, i_dt, derivative, inputs, kStampOffset);
    auto result = Result<Output>{derivative, jacobians, layout.outer_input_size, StampedOutputTangent::kNumParameters};
    SpatialInterpolator<TOutput, TInput>::evaluate(result, weights, inputs, s_idx, e_idx, kVariableOffset);
    return result;
  }
}

template <typename TOutput, typename TInput>
auto ContinuousState<TOutput, TInput>::iterators(const Time& time) const -> std::tuple<Iterator, Iterator, Index> {
  DCHECK(range().contains(time)) << "Range does not contain time.";
  const auto layout = interpolator()->layout(this->isUniform());

  DCHECK_LE(layout.outer_input_size, this->stamped_inputs_.size());
  const auto itr = this->stamped_inputs_.upper_bound(time);
  const auto begin = std::prev(itr, layout.left_input_margin);
  const auto end = std::next(itr, layout.right_input_margin);
  return {begin, end, layout.outer_input_size};
}

template class ContinuousState<Cartesian<double, 3>>;
template class ContinuousState<SU2<double>>;
template class ContinuousState<SE3<double>>;

}  // namespace hyper::state
