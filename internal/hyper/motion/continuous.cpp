/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <iostream>

#include <glog/logging.h>

#include "hyper/matrix.hpp"
#include "hyper/motion/continuous.hpp"
#include "hyper/variables/groups/se3.hpp"
#include "hyper/variables/stamped.hpp"

namespace hyper {

template <typename TVariable>
ContinuousMotion<TVariable>::ContinuousMotion() = default;

template <typename TVariable>
ContinuousMotion<TVariable>::ContinuousMotion(const TemporalInterpolator<Scalar>* interpolator) {
  setInterpolator(interpolator);
}

template <typename TVariable>
auto ContinuousMotion<TVariable>::range() const -> Range {
  const auto layout = interpolator()->layout();
  DCHECK_LE(layout.outer_input_size, this->elements_.size());
  const auto v0_itr = std::next(this->elements_.cbegin(), layout.left_input_margin - 1);
  const auto vn_itr = std::next(this->elements_.crbegin(), layout.right_input_margin - 1);
  DCHECK_LT(v0_itr->stamp(), vn_itr->stamp());
  return {v0_itr->stamp(), vn_itr->stamp()};
}

template <typename TVariable>
auto ContinuousMotion<TVariable>::pointers(const Time& time) const -> Pointers<Element> {
  const auto& [begin, end, num_inputs] = iterators(time);
  Pointers<Element> pointers;
  pointers.reserve(num_inputs);
  std::transform(begin, end, std::back_inserter(pointers), [](const auto& element) { return const_cast<Element*>(&element); });
  DCHECK_EQ(pointers.size(), num_inputs);
  return pointers;
}

template <typename TVariable>
auto ContinuousMotion<TVariable>::interpolator() const -> const TemporalInterpolator<Scalar>* {
  return interpolator_;
}

template <typename TVariable>
auto ContinuousMotion<TVariable>::setInterpolator(const TemporalInterpolator<Scalar>* interpolator) -> void {
  DCHECK(interpolator != nullptr);
  interpolator_ = interpolator;
}

template <typename TVariable>
auto ContinuousMotion<TVariable>::evaluate(const Time& time, const Derivative& derivative, bool jacobians) const -> TemporalMotionResult<TVariable> {
  const auto& [begin, end, num_inputs] = iterators(time);
  Pointers<const Scalar> pointers;
  pointers.reserve(num_inputs);
  std::transform(begin, end, std::back_inserter(pointers), [](const auto& element) { return element.data(); });
  DCHECK_EQ(pointers.size(), num_inputs);
  return evaluate(time, derivative, jacobians, pointers.data());
}

template <typename TVariable>
auto ContinuousMotion<TVariable>::evaluate(const Time& time, const Derivative& derivative, bool jacobians, const Scalar* const* elements) const -> TemporalMotionResult<TVariable> {
  // Definitions.
  using Stamps = std::vector<Scalar>;

  // Fetch layout.
  const auto layout = interpolator()->layout();

  // Split pointers.
  Pointers<const Scalar> variables;
  variables.reserve(layout.outer_input_size);

  Stamps stamps;
  stamps.reserve(layout.outer_input_size);

  for (Index i = 0; i < layout.outer_input_size; ++i) {
    const auto p_element_i = elements[i];
    stamps.emplace_back(p_element_i[Element::kStampOffset]);
    variables.emplace_back(p_element_i + Element::kVariableOffset);
  }

  const auto offset = layout.left_input_margin - 1;
  const auto weights = interpolator()->evaluate(time, derivative, stamps, offset);

  auto result = TemporalMotionResult<TVariable>{derivative, layout.outer_input_size, jacobians};
  SpatialInterpolator<TVariable>::evaluate(weights, variables, result.outputs, jacobians ? &result.jacobians : nullptr, layout.left_input_padding, Element::kNumParameters);
  return result;
}

template <typename TVariable>
auto ContinuousMotion<TVariable>::iterators(const Time& time) const -> std::tuple<Iterator, Iterator, Index> {
  DCHECK(range().contains(time)) << "Range does not contain time.";
  const auto layout = interpolator()->layout();

  DCHECK_LE(layout.outer_input_size, this->elements_.size());
  const auto itr = this->elements_.upper_bound(time);
  const auto begin = std::prev(itr, layout.left_input_margin);
  const auto end = std::next(itr, layout.right_input_margin);
  return {begin, end, layout.outer_input_size};
}

template class ContinuousMotion<Cartesian<double, 3>>;
template class ContinuousMotion<SE3<double>>;

} // namespace hyper
