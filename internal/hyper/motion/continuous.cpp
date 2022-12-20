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
  if (temporal_interpolator_) {
    const auto layout = temporal_interpolator_->layout();
    DCHECK_LE(layout.outer_input_size, this->elements_.size());
    const auto& v0 = *std::next(this->elements_.cbegin(), layout.left_input_margin - 1);
    const auto& vn = *std::next(this->elements_.crbegin(), layout.right_input_margin - 1);
    DCHECK_LT(v0.stamp(), vn.stamp());
    return {v0.stamp(), vn.stamp()};
  } else {
    DCHECK(!this->elements_.empty());
    const auto& v0 = *this->elements_.cbegin();
    const auto& vn = *this->elements_.crbegin();
    return {v0.stamp(), std::nextafter(vn.stamp(), std::numeric_limits<Scalar>::infinity())};
  }
}

template <typename TVariable>
auto ContinuousMotion<TVariable>::pointers(const Time& time) const -> Pointers<Element> {
  DCHECK(range().contains(time)) << "State range does not contain stamp.";
  if (temporal_interpolator_) {
    const auto layout = temporal_interpolator_->layout();
    DCHECK_LE(layout.outer_input_size, this->elements_.size());

    const auto itr = this->elements_.upper_bound(time);
    const auto begin = std::prev(itr, layout.left_input_margin);
    const auto end = std::next(itr, layout.right_input_margin);

    Pointers<Element> pointers;
    pointers.reserve(layout.outer_input_size);
    std::transform(begin, end, std::back_inserter(pointers), [](const auto& arg) { return const_cast<Element*>(&arg); });
    DCHECK_EQ(pointers.size(), layout.outer_input_size);
    return pointers;

  } else {
    const auto itr = this->elements_.find(time);
    DCHECK(itr != this->elements_.cend()) << "State does not contain stamp.";
    return {const_cast<Element*>(&(*itr))};
  }
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
  Pointers<const Scalar> pointers;
  const auto elements = this->pointers(query.time);
  pointers.reserve(elements.size());
  std::transform(elements.begin(), elements.end(), std::back_inserter(pointers), [](const auto& arg) { return arg->asVector().data(); });
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

template class ContinuousMotion<Cartesian<double, 3>>;
template class ContinuousMotion<SE3<double>>;

} // namespace hyper
