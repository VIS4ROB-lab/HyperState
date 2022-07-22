/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <glog/logging.h>

#include "hyper/state/abstract.hpp"

namespace hyper {

AbstractState::AbstractState(std::unique_ptr<AbstractInterpolator>&& interpolator, std::unique_ptr<AbstractPolicy>&& policy)
    : variables_{},
      interpolator_{std::move(interpolator)},
      policy_{std::move(policy)} {}

auto AbstractState::variables() const -> const Variables& {
  return variables_;
}

auto AbstractState::variables() -> Variables& {
  return const_cast<Variables&>(std::as_const(*this).variables());
}

auto AbstractState::memoryBlocks() const -> MemoryBlocks<Scalar> {
  MemoryBlocks<Scalar> memory_blocks;
  memory_blocks.reserve(variables_.size());
  std::transform(variables_.begin(), variables_.end(), std::back_inserter(memory_blocks), [](const auto& arg) { return arg->memory(); });
  return memory_blocks;
}

auto AbstractState::memoryBlocks(const Stamp& stamp) const -> MemoryBlocks<Scalar> {
  DCHECK(range().contains(stamp)) << "State range does not contain stamp.";
  if (interpolator_) {
    const auto outer = interpolator_->layout().outer;
    const auto num_variables = outer.size() + 1;
    DCHECK_LE(num_variables, variables_.size());

    const auto itr = variables_.upper_bound(stamp);
    const auto begin = std::next(itr, outer.lower - 1);
    const auto end = std::next(itr, outer.upper - 1);

    MemoryBlocks<Scalar> memory_blocks;
    memory_blocks.reserve(num_variables);
    std::transform(begin, end, std::back_inserter(memory_blocks), [](const auto& arg) { return arg->memory(); });
    return memory_blocks;

  } else {
    const auto itr = variables_.find(stamp);
    DCHECK(itr != variables_.cend()) << "State does not contain stamp.";
    return {(*itr)->memory()};
  }
}

auto AbstractState::range() const -> Range {
  if (interpolator_) {
    const auto outer = interpolator_->layout().outer;
    const auto num_variables = outer.size() + 1;
    DCHECK_LE(num_variables, variables_.size());
    const auto& v0 = *std::next(variables_.cbegin(), -outer.lowerBound());
    const auto& vn = *std::next(variables_.cend(), -outer.upperBound());
    DCHECK_LT(v0->stamp(), vn->stamp());
    return {v0->stamp(), vn->stamp()};
  } else {
    DCHECK(!variables_.empty());
    const auto& v0 = *variables_.cbegin();
    const auto& vn = *variables_.crbegin();
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
    const auto pointers = memoryBlocks(state_query.stamp).addresses<const Scalar>();
    const auto stamps = policy_->stamps(pointers);
    const auto weights = interpolator_->weights(state_query, stamps);
    const auto policy_query = PolicyQuery{layout, pointers, weights};
    return policy_->evaluate(state_query, policy_query);
  } else {
    const auto pointers = memoryBlocks(state_query.stamp).addresses<const Scalar>();
    const auto policy_query = PolicyQuery{{}, pointers, {}};
    return policy_->evaluate(state_query, policy_query);
  }
}

auto AbstractState::evaluate(const StateQuery& state_query, const Scalar* const* raw_values) const -> StateResult {
  DCHECK(policy_ != nullptr);
  if (interpolator_) {
    const auto layout = interpolator_->layout();
    const auto pointers = Pointers<const Scalar>{raw_values, raw_values + layout.outer.size() + 1};
    const auto stamps = policy_->stamps(pointers);
    const auto weights = interpolator_->weights(state_query, stamps);
    const auto policy_query = PolicyQuery{layout, pointers, weights};
    return policy_->evaluate(state_query, policy_query);
  } else {
    const auto pointers = Pointers<const Scalar>{raw_values, raw_values + 1};
    const auto policy_query = PolicyQuery{{}, pointers, {}};
    return policy_->evaluate(state_query, policy_query);
  }
}

} // namespace hyper
