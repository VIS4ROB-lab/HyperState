/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <iostream>

#include <glog/logging.h>

#include "hyper/matrix.hpp"
#include "hyper/state/continuous.hpp"
#include "hyper/variables/groups/se3.hpp"
#include "hyper/variables/stamped.hpp"

namespace hyper::state {

template <typename TVariable>
ContinuousState<TVariable>::ContinuousState() = default;

template <typename TVariable>
ContinuousState<TVariable>::ContinuousState(const TemporalInterpolator<Scalar>* interpolator) {
  setInterpolator(interpolator);
}

template <typename TVariable>
auto ContinuousState<TVariable>::range() const -> Range {
  const auto layout = interpolator()->layout();
  DCHECK_LE(layout.outer_input_size, this->stamped_variables_.size());
  const auto v0_itr = std::next(this->stamped_variables_.cbegin(), layout.left_input_margin - 1);
  const auto vn_itr = std::next(this->stamped_variables_.crbegin(), layout.right_input_margin - 1);
  DCHECK_LT(v0_itr->stamp(), vn_itr->stamp());
  return {v0_itr->stamp(), vn_itr->stamp()};
}

template <typename TVariable>
auto ContinuousState<TVariable>::variables() const -> std::vector<StampedVariable*> {
  std::vector<StampedVariable*> ptrs;
  ptrs.reserve(this->stamped_variables_.size());
  std::transform(this->stamped_variables_.begin(), this->stamped_variables_.end(), std::back_inserter(ptrs),
                 [](const auto& element) { return const_cast<StampedVariable*>(&element); });
  return ptrs;
}

template <typename TVariable>
auto ContinuousState<TVariable>::variables(const Time& time) const -> std::vector<StampedVariable*> {
  const auto& [begin, end, num_inputs] = iterators(time);
  std::vector<StampedVariable*> ptrs;
  ptrs.reserve(num_inputs);
  std::transform(begin, end, std::back_inserter(ptrs), [](const auto& element) { return const_cast<StampedVariable*>(&element); });
  return ptrs;
}

template <typename TVariable>
auto ContinuousState<TVariable>::parameterBlocks() const -> std::vector<Scalar*> {
  std::vector<Scalar*> ptrs;
  ptrs.reserve(this->stamped_variables_.size());
  std::transform(this->stamped_variables_.begin(), this->stamped_variables_.end(), std::back_inserter(ptrs),
                 [](const auto& element) { return const_cast<Scalar*>(element.data()); });
  return ptrs;
}

template <typename TVariable>
auto ContinuousState<TVariable>::parameterBlocks(const Time& time) const -> std::vector<Scalar*> {
  const auto& [begin, end, num_inputs] = iterators(time);
  std::vector<Scalar*> ptrs;
  ptrs.reserve(num_inputs);
  std::transform(begin, end, std::back_inserter(ptrs), [](const auto& element) { return const_cast<Scalar*>(element.data()); });
  return ptrs;
}

template <typename TVariable>
auto ContinuousState<TVariable>::interpolator() const -> const TemporalInterpolator<Scalar>* {
  return interpolator_;
}

template <typename TVariable>
auto ContinuousState<TVariable>::setInterpolator(const TemporalInterpolator<Scalar>* interpolator) -> void {
  DCHECK(interpolator != nullptr);
  interpolator_ = interpolator;
}

template <typename TVariable>
auto ContinuousState<TVariable>::evaluate(const Time& time, const Index& derivative, bool jacobians) const -> Result<TVariable> {
  const auto& [begin, end, num_inputs] = iterators(time);
  std::vector<const Scalar*> pointers;
  pointers.reserve(num_inputs);
  std::transform(begin, end, std::back_inserter(pointers), [](const auto& element) { return element.data(); });
  DCHECK_EQ(pointers.size(), num_inputs);
  return evaluate(time, derivative, jacobians, pointers.data());
}

template <typename TVariable>
auto ContinuousState<TVariable>::evaluate(const Time& time, const Index& derivative, bool jacobians, const Scalar* const* elements) const -> Result<TVariable> {
  // Definitions.
  using Stamps = std::vector<Scalar>;

  // Fetch layout.
  const auto layout = interpolator()->layout();

  // Split pointers.
  Stamps stamps;
  std::vector<const Scalar*> inputs;
  inputs.reserve(layout.outer_input_size);
  stamps.reserve(layout.outer_input_size);

  for (Index i = 0; i < layout.outer_input_size; ++i) {
    const auto p_element_i = elements[i];
    inputs.emplace_back(p_element_i + StampedVariable::kVariableOffset);
    stamps.emplace_back(p_element_i[StampedVariable::kStampOffset]);
  }

  const auto offset = layout.left_input_margin - 1;
  const auto weights = interpolator()->evaluate(time, derivative, stamps, offset);

  return SpatialInterpolator<TVariable>::evaluate(inputs, weights, jacobians, layout.left_input_padding, StampedVariable::kNumParameters);
}

template <typename TVariable>
auto ContinuousState<TVariable>::iterators(const Time& time) const -> std::tuple<Iterator, Iterator, Index> {
  DCHECK(range().contains(time)) << "Range does not contain time.";
  const auto layout = interpolator()->layout();

  DCHECK_LE(layout.outer_input_size, this->stamped_variables_.size());
  const auto itr = this->stamped_variables_.upper_bound(time);
  const auto begin = std::prev(itr, layout.left_input_margin);
  const auto end = std::next(itr, layout.right_input_margin);
  return {begin, end, layout.outer_input_size};
}

template class ContinuousState<variables::Cartesian<double, 3>>;
template class ContinuousState<variables::SE3<double>>;

}  // namespace hyper::state
