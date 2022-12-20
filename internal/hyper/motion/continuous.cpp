/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <glog/logging.h>

#include "hyper/matrix.hpp"
#include "hyper/motion/continuous.hpp"
#include "hyper/variables/groups/se3.hpp"
#include "hyper/variables/stamped.hpp"

namespace hyper {

template <typename TVariable>
ContinuousMotion<TVariable>::ContinuousMotion(std::unique_ptr<TemporalInterpolator<Scalar>>&& temporal_interpolator)
    : temporal_interpolator_{std::move(temporal_interpolator)} {}

template <typename TVariable>
auto ContinuousMotion<TVariable>::range() const -> Range {
  DCHECK(temporal_interpolator_ != nullptr);
  const auto layout = temporal_interpolator_->layout();
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
auto ContinuousMotion<TVariable>::interpolator() const -> const std::unique_ptr<TemporalInterpolator<Scalar>>& {
  return temporal_interpolator_;
}

template <typename TVariable>
auto ContinuousMotion<TVariable>::interpolator() -> std::unique_ptr<TemporalInterpolator<Scalar>>& {
  return const_cast<std::unique_ptr<TemporalInterpolator<Scalar>>&>(std::as_const(*this).interpolator());
}

template <typename TVariable>
auto ContinuousMotion<TVariable>::evaluate(const Query& query) const -> bool {
  const auto& [begin, end, num_inputs] = iterators(query.time);
  Pointers<const Scalar> pointers;
  pointers.reserve(num_inputs);
  std::transform(begin, end, std::back_inserter(pointers), [](const auto& element) { return element.data(); });
  DCHECK_EQ(pointers.size(), num_inputs);
  return evaluate(query, pointers.data());
}

template <typename TVariable>
auto ContinuousMotion<TVariable>::evaluate(const Query& query, const Scalar* const* inputs) const -> bool {
  DCHECK(temporal_interpolator_ != nullptr);
  const auto layout = temporal_interpolator_->layout();
  const auto timestamps = this->extractTimestamps(inputs, layout.outer_input_size);
  const auto weights = temporal_interpolator_->evaluate(query.time, query.derivative, layout.left_input_margin - 1, timestamps);
  return SpatialInterpolator<Element>::evaluate(query, layout, weights, inputs);
}

template <typename TVariable>
auto ContinuousMotion<TVariable>::iterators(const Time& time) const -> std::tuple<Iterator, Iterator, Index> {
  DCHECK(range().contains(time)) << "State range does not contain stamp.";
  const auto layout = temporal_interpolator_->layout();

  DCHECK_LE(layout.outer_input_size, this->elements_.size());
  const auto itr = this->elements_.upper_bound(time);
  const auto begin = std::prev(itr, layout.left_input_margin);
  const auto end = std::next(itr, layout.right_input_margin);
  return {begin, end, layout.outer_input_size};
}

template class ContinuousMotion<Cartesian<double, 3>>;
template class ContinuousMotion<SE3<double>>;

} // namespace hyper
