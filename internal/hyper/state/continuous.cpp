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
  DCHECK_LE(layout.outer_input_size, this->elements_.size());
  const auto v0_itr = std::next(this->elements_.cbegin(), layout.left_input_margin - 1);
  const auto vn_itr = std::next(this->elements_.crbegin(), layout.right_input_margin - 1);
  DCHECK_LT(v0_itr->stamp(), vn_itr->stamp());
  return {v0_itr->stamp(), vn_itr->stamp()};
}

template <typename TVariable>
auto ContinuousState<TVariable>::pointers() const -> std::vector<Element*> {
  std::vector<Element*> pointers;
  pointers.reserve(this->elements_.size());
  std::transform(this->elements_.begin(), this->elements_.end(), std::back_inserter(pointers), [](const auto& element) { return const_cast<Element*>(&element); });
  return pointers;
}

template <typename TVariable>
auto ContinuousState<TVariable>::pointers(const Time& time) const -> std::vector<Element*> {
  const auto& [begin, end, num_inputs] = iterators(time);
  std::vector<Element*> pointers;
  pointers.reserve(num_inputs);
  std::transform(begin, end, std::back_inserter(pointers), [](const auto& element) { return const_cast<Element*>(&element); });
  return pointers;
}

template <typename TVariable>
auto ContinuousState<TVariable>::parameters() const -> std::vector<Scalar*> {
  std::vector<Scalar*> parameters;
  parameters.reserve(this->elements_.size());
  std::transform(this->elements_.begin(), this->elements_.end(), std::back_inserter(parameters), [](const auto& element) { return const_cast<Scalar*>(element.data()); });
  return parameters;
}

template <typename TVariable>
auto ContinuousState<TVariable>::parameters(const Time& time) const -> std::vector<Scalar*> {
  const auto& [begin, end, num_inputs] = iterators(time);
  std::vector<Scalar*> parameters;
  parameters.reserve(num_inputs);
  std::transform(begin, end, std::back_inserter(parameters), [](const auto& element) { return const_cast<Scalar*>(element.data()); });
  return parameters;
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
  std::vector<const Scalar*> variables;
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

  auto result = Result<TVariable>{derivative, layout.outer_input_size, jacobians};
  SpatialInterpolator<TVariable>::evaluate(variables, weights, result.outputs, jacobians ? &result.jacobians : nullptr, layout.left_input_padding, Element::kNumParameters);
  return result;
}

template <typename TVariable>
auto ContinuousState<TVariable>::iterators(const Time& time) const -> std::tuple<Iterator, Iterator, Index> {
  DCHECK(range().contains(time)) << "Range does not contain time.";
  const auto layout = interpolator()->layout();

  DCHECK_LE(layout.outer_input_size, this->elements_.size());
  const auto itr = this->elements_.upper_bound(time);
  const auto begin = std::prev(itr, layout.left_input_margin);
  const auto end = std::next(itr, layout.right_input_margin);
  return {begin, end, layout.outer_input_size};
}

template class ContinuousState<variables::Cartesian<double, 3>>;
template class ContinuousState<variables::SE3<double>>;

}  // namespace hyper::state
