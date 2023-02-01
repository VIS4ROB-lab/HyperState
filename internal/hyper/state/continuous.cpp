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

template <typename TOutput, typename TVariable>
ContinuousState<TOutput, TVariable>::ContinuousState() = default;

template <typename TOutput, typename TVariable>
ContinuousState<TOutput, TVariable>::ContinuousState(const TemporalInterpolator<Scalar>* interpolator) {
  setInterpolator(interpolator);
}

template <typename TOutput, typename TVariable>
auto ContinuousState<TOutput, TVariable>::range() const -> Range {
  const auto layout = interpolator()->layout();
  DCHECK_LE(layout.outer_input_size, this->stamped_variables_.size());
  const auto v0_itr = std::next(this->stamped_variables_.cbegin(), layout.left_input_margin - 1);
  const auto vn_itr = std::next(this->stamped_variables_.crbegin(), layout.right_input_margin - 1);
  DCHECK_LT(v0_itr->stamp(), vn_itr->stamp());
  return {v0_itr->stamp(), vn_itr->stamp()};
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
  const auto& [begin, end, num_inputs] = iterators(time);
  std::vector<StampedVariable*> ptrs;
  ptrs.reserve(num_inputs);
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
  const auto& [begin, end, num_inputs] = iterators(time);
  std::vector<Scalar*> ptrs;
  ptrs.reserve(num_inputs);
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
auto ContinuousState<TOutput, TVariable>::evaluate(const Time& time, const Index& derivative, bool jacobians) const -> Result<TOutput> {
  const auto& [begin, end, num_inputs] = iterators(time);
  std::vector<const Scalar*> ptrs;
  ptrs.reserve(num_inputs);
  std::transform(begin, end, std::back_inserter(ptrs), [](const auto& element) { return element.data(); });
  DCHECK_EQ(ptrs.size(), num_inputs);
  return evaluate(time, derivative, jacobians, ptrs.data());
}

template <typename TOutput, typename TVariable>
auto ContinuousState<TOutput, TVariable>::evaluate(const Time& time, const Index& derivative, bool jacobians, const Scalar* const* inputs) const -> Result<TOutput> {
  // Fetch layout.
  const auto layout = interpolator()->layout();

  // Split pointers.
  std::vector<Time> times;
  times.reserve(layout.outer_input_size);

  for (Index i = 0; i < layout.outer_input_size; ++i) {
    times.emplace_back(inputs[i][StampedVariable::kStampOffset]);
  }

  const auto s_idx = layout.left_input_padding;
  const auto e_idx = layout.left_input_padding + layout.inner_input_size - 1;
  auto result = Result<Output>{derivative, jacobians, layout.outer_input_size, Stamped<Tangent<Output>>::kNumParameters};
  const auto weights = interpolator()->evaluate(time, derivative, times, layout.left_input_margin - 1);
  SpatialInterpolator<TOutput, TVariable>::evaluate(result, weights, inputs, s_idx, e_idx, StampedVariable::kVariableOffset);
  return result;
}

template <typename TOutput, typename TVariable>
auto ContinuousState<TOutput, TVariable>::iterators(const Time& time) const -> std::tuple<Iterator, Iterator, Index> {
  DCHECK(range().contains(time)) << "Range does not contain time.";
  const auto layout = interpolator()->layout();

  DCHECK_LE(layout.outer_input_size, this->stamped_variables_.size());
  const auto itr = this->stamped_variables_.upper_bound(time);
  const auto begin = std::prev(itr, layout.left_input_margin);
  const auto end = std::next(itr, layout.right_input_margin);
  return {begin, end, layout.outer_input_size};
}

template class ContinuousState<Cartesian<double, 3>>;
template class ContinuousState<SU2<double>>;
template class ContinuousState<SE3<double>>;
template class ContinuousState<SE3<double>, Tangent<SE3<double>>>;

}  // namespace hyper::state
