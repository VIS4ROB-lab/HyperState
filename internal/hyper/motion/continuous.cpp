/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <glog/logging.h>

#include "hyper/matrix.hpp"
#include "hyper/motion/continuous.hpp"
#include "hyper/variables/groups/se3.hpp"
#include "hyper/variables/stamped.hpp"

namespace hyper {

namespace {

/// Converts parameter pointers to raw pointers.
/// \tparam TScalar Scalar type.
/// \tparam TParameter Parameter type.
/// \param parameters Input parameters.
/// \return Converted pointers
template <typename TScalar, typename TParameter>
auto convertPointers(const Pointers<TParameter>& parameters) -> Pointers<TScalar> {
  Pointers<TScalar> pointers;
  pointers.reserve(parameters.size());
  std::transform(parameters.begin(), parameters.end(), std::back_inserter(pointers), [](const auto& arg) { return arg->asVector().data(); });
  return pointers;
}

} // namespace

template <typename TVariable>
ContinuousMotion<TVariable>::ContinuousMotion(std::unique_ptr<TemporalInterpolator<Scalar>>&& interpolator)
    : interpolator_{std::move(interpolator)} {}

template <typename TVariable>
auto ContinuousMotion<TVariable>::range() const -> Range {
  if (interpolator_) {
    const auto layout = interpolator_->layout();
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
  if (interpolator_) {
    const auto layout = interpolator_->layout();
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
  return interpolator_;
}

template <typename TVariable>
auto ContinuousMotion<TVariable>::interpolator() -> std::unique_ptr<TemporalInterpolator<Scalar>>& {
  return const_cast<std::unique_ptr<TemporalInterpolator<Scalar>>&>(std::as_const(*this).interpolator());
}

template <typename TVariable>
auto ContinuousMotion<TVariable>::evaluate(const Query& query) const -> bool {
  DCHECK(false);
  return false;
}

template <typename TVariable>
auto ContinuousMotion<TVariable>::evaluate(const Query& query, const Scalar* const* pointers) const -> bool {
  DCHECK(false);
  return false;
}

template <typename TVariable>
auto ContinuousMotion<TVariable>::evaluate(const StateQuery& state_query) const -> StateResult {
  if (interpolator_) {
    const auto layout = interpolator_->layout();
    const auto pointers = convertPointers<const Scalar>(this->pointers(state_query.time));
    const auto stamps = this->extractTimes(pointers.data(), layout.outer_input_size);

    MatrixX<Scalar> weights{layout.output_size, state_query.derivative + 1};
    interpolator_->evaluate({state_query.time, state_query.derivative, stamps, weights.data()});

    const auto policy_query = PolicyQuery{layout, pointers, weights};
    return SpatialInterpolator<Element>::evaluate(state_query, policy_query);
  } else {
    const auto pointers = convertPointers<const Scalar>(this->pointers(state_query.time));
    const auto policy_query = PolicyQuery{{}, pointers, {}};
    return SpatialInterpolator<Element>::evaluate(state_query, policy_query);
  }
}

template <typename TVariable>
auto ContinuousMotion<TVariable>::evaluate(const StateQuery& state_query, const Scalar* const* raw_values) const -> StateResult {
  if (interpolator_) {
    const auto layout = interpolator_->layout();
    const auto pointers = Pointers<const Scalar>{raw_values, raw_values + layout.outer_input_size};
    const auto stamps = this->extractTimes(pointers.data(), layout.outer_input_size);

    MatrixX<Scalar> weights{layout.output_size, state_query.derivative + 1};
    interpolator_->evaluate({state_query.time, state_query.derivative, stamps, weights.data()});

    const auto policy_query = PolicyQuery{layout, pointers, weights};
    return SpatialInterpolator<Element>::evaluate(state_query, policy_query);
  } else {
    const auto pointers = Pointers<const Scalar>{raw_values, raw_values + 1};
    const auto policy_query = PolicyQuery{{}, pointers, {}};
    return SpatialInterpolator<Element>::evaluate(state_query, policy_query);
  }
}

template class ContinuousMotion<Cartesian<double, 3>>;
template class ContinuousMotion<SE3<double>>;

} // namespace hyper
