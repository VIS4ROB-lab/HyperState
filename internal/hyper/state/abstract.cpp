/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <glog/logging.h>

#include "hyper/state/abstract.hpp"

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

AbstractState::AbstractState(std::unique_ptr<AbstractInterpolator>&& interpolator, std::unique_ptr<AbstractPolicy>&& policy)
    : elements_{},
      interpolator_{std::move(interpolator)},
      policy_{std::move(policy)} {}

auto AbstractState::elements() const -> const Elements& {
  return elements_;
}

auto AbstractState::elements() -> Elements& {
  return const_cast<Elements&>(std::as_const(*this).elements());
}

auto AbstractState::parameters() const -> Pointers<Parameter> {
  Pointers<Parameter> pointers;
  pointers.reserve(elements_.size());
  std::transform(elements_.begin(), elements_.end(), std::back_inserter(pointers), [](const auto& arg) { return arg.get(); });
  return pointers;
}

auto AbstractState::parameters(const Time& time) const -> Pointers<Parameter> {
  DCHECK(range().contains(time)) << "State range does not contain stamp.";
  if (interpolator_) {
    const auto layout = interpolator_->layout();
    DCHECK_LE(layout.outer_input_size, elements_.size());

    const auto itr = elements_.upper_bound(time);
    const auto begin = std::prev(itr, layout.left_input_margin);
    const auto end = std::next(itr, layout.right_input_margin);

    Pointers<Parameter> pointers;
    pointers.reserve(layout.outer_input_size);
    std::transform(begin, end, std::back_inserter(pointers), [](const auto& arg) { return arg.get(); });
    DCHECK_EQ(pointers.size(), layout.outer_input_size);
    return pointers;

  } else {
    const auto itr = elements_.find(time);
    DCHECK(itr != elements_.cend()) << "State does not contain stamp.";
    return {itr->get()};
  }
}

auto AbstractState::range() const -> Range {
  if (interpolator_) {
    const auto layout = interpolator_->layout();
    DCHECK_LE(layout.outer_input_size, elements_.size());
    const auto& v0 = *std::next(elements_.cbegin(), layout.left_input_margin - 1);
    const auto& vn = *std::next(elements_.crbegin(), layout.right_input_margin - 1);
    DCHECK_LT(v0->stamp(), vn->stamp());
    return {v0->stamp(), vn->stamp()};
  } else {
    DCHECK(!elements_.empty());
    const auto& v0 = *elements_.cbegin();
    const auto& vn = *elements_.crbegin();
    return {v0->stamp(), std::nextafter(vn->stamp(), std::numeric_limits<Scalar>::infinity())};
  }
}

auto AbstractState::interpolator() const -> const std::unique_ptr<AbstractInterpolator>& {
  return interpolator_;
}

auto AbstractState::interpolator() -> std::unique_ptr<AbstractInterpolator>& {
  return const_cast<std::unique_ptr<AbstractInterpolator>&>(std::as_const(*this).interpolator());
}

auto AbstractState::policy() const -> const std::unique_ptr<AbstractPolicy>& {
  return policy_;
}

auto AbstractState::policy() -> std::unique_ptr<AbstractPolicy>& {
  return const_cast<std::unique_ptr<AbstractPolicy>&>(std::as_const(*this).policy());
}

auto AbstractState::evaluate(const StateQuery& state_query) const -> StateResult {
  DCHECK(policy_ != nullptr);
  if (interpolator_) {
    const auto layout = interpolator_->layout();
    const auto pointers = convertPointers<const Scalar>(parameters(state_query.time));
    const auto stamps = policy_->times(pointers);
    const auto weights = interpolator_->weights(state_query.time, stamps, state_query.derivative);
    const auto policy_query = PolicyQuery{layout, pointers, weights};
    return policy_->evaluate(state_query, policy_query);
  } else {
    const auto pointers = convertPointers<const Scalar>(parameters(state_query.time));
    const auto policy_query = PolicyQuery{{}, pointers, {}};
    return policy_->evaluate(state_query, policy_query);
  }
}

auto AbstractState::evaluate(const StateQuery& state_query, const Scalar* const* raw_values) const -> StateResult {
  DCHECK(policy_ != nullptr);
  if (interpolator_) {
    const auto layout = interpolator_->layout();
    const auto pointers = Pointers<const Scalar>{raw_values, raw_values + layout.outer_input_size};
    const auto stamps = policy_->times(pointers);
    const auto weights = interpolator_->weights(state_query.time, stamps, state_query.derivative);
    const auto policy_query = PolicyQuery{layout, pointers, weights};
    return policy_->evaluate(state_query, policy_query);
  } else {
    const auto pointers = Pointers<const Scalar>{raw_values, raw_values + 1};
    const auto policy_query = PolicyQuery{{}, pointers, {}};
    return policy_->evaluate(state_query, policy_query);
  }
}

} // namespace hyper
