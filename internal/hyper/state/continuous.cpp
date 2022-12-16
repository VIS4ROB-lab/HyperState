/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <glog/logging.h>

#include "hyper/matrix.hpp"
#include "hyper/state/continuous.hpp"

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

ContinuousMotion::ContinuousMotion(std::unique_ptr<TemporalInterpolator<Scalar>>&& interpolator, std::unique_ptr<AbstractPolicy>&& policy)
    : elements_{},
      interpolator_{std::move(interpolator)},
      policy_{std::move(policy)} {}

auto ContinuousMotion::elements() const -> const Elements& {
  return elements_;
}

auto ContinuousMotion::elements() -> Elements& {
  return const_cast<Elements&>(std::as_const(*this).elements());
}

auto ContinuousMotion::parameters() const -> Pointers<Parameter> {
  Pointers<Parameter> pointers;
  pointers.reserve(elements_.size());
  std::transform(elements_.begin(), elements_.end(), std::back_inserter(pointers), [](const auto& arg) { return arg.get(); });
  return pointers;
}

auto ContinuousMotion::parameters(const Time& time) const -> Pointers<Parameter> {
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

auto ContinuousMotion::range() const -> Range {
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

auto ContinuousMotion::interpolator() const -> const std::unique_ptr<TemporalInterpolator<Scalar>>& {
  return interpolator_;
}

auto ContinuousMotion::interpolator() -> std::unique_ptr<TemporalInterpolator<Scalar>>& {
  return const_cast<std::unique_ptr<TemporalInterpolator<Scalar>>&>(std::as_const(*this).interpolator());
}

auto ContinuousMotion::policy() const -> const std::unique_ptr<AbstractPolicy>& {
  return policy_;
}

auto ContinuousMotion::policy() -> std::unique_ptr<AbstractPolicy>& {
  return const_cast<std::unique_ptr<AbstractPolicy>&>(std::as_const(*this).policy());
}

auto ContinuousMotion::evaluate(const StateQuery& state_query) const -> StateResult {
  DCHECK(policy_ != nullptr);
  if (interpolator_) {
    const auto layout = interpolator_->layout();
    const auto pointers = convertPointers<const Scalar>(parameters(state_query.time));
    const auto stamps = policy_->times(pointers);

    MatrixX<Scalar> weights{layout.output_size, state_query.derivative + 1};
    interpolator_->evaluate({state_query.time, state_query.derivative, stamps, weights.data()});

    const auto policy_query = PolicyQuery{layout, pointers, weights};
    return policy_->evaluate(state_query, policy_query);
  } else {
    const auto pointers = convertPointers<const Scalar>(parameters(state_query.time));
    const auto policy_query = PolicyQuery{{}, pointers, {}};
    return policy_->evaluate(state_query, policy_query);
  }
}

auto ContinuousMotion::evaluate(const StateQuery& state_query, const Scalar* const* raw_values) const -> StateResult {
  DCHECK(policy_ != nullptr);
  if (interpolator_) {
    const auto layout = interpolator_->layout();
    const auto pointers = Pointers<const Scalar>{raw_values, raw_values + layout.outer_input_size};
    const auto stamps = policy_->times(pointers);

    MatrixX<Scalar> weights{layout.output_size, state_query.derivative + 1};
    interpolator_->evaluate({state_query.time, state_query.derivative, stamps, weights.data()});

    const auto policy_query = PolicyQuery{layout, pointers, weights};
    return policy_->evaluate(state_query, policy_query);
  } else {
    const auto pointers = Pointers<const Scalar>{raw_values, raw_values + 1};
    const auto policy_query = PolicyQuery{{}, pointers, {}};
    return policy_->evaluate(state_query, policy_query);
  }
}

} // namespace hyper