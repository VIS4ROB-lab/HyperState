/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <memory>
#include <set>

#include <glog/logging.h>

#include "hyper/state/state.hpp"
#include "hyper/variables/jacobian.hpp"
#include "hyper/variables/stamped.hpp"
#include "hyper/variables/variable.hpp"
#include "range.hpp"

namespace hyper::state {

template <typename TOutput, typename TVariable>
class TemporalState : public State<typename TVariable::Scalar> {
 public:
  // Definitions.
  using Base = State<typename TVariable::Scalar>;

  using Index = typename Base::Index;
  using Scalar = typename Base::Scalar;

  using Time = Scalar;
  using Range = state::Range<Time, BoundaryPolicy::INCLUSIVE>;

  using StampedVariable = variables::Stamped<TVariable>;

  // Stamped variable compare.
  struct StampedVariableCompare {
    using is_transparent = std::true_type;
    auto operator()(const StampedVariable& lhs, const StampedVariable& rhs) const -> bool { return lhs.stamp() < rhs.stamp(); }
    auto operator()(const StampedVariable& lhs, const Time& rhs) const -> bool { return lhs.stamp() < rhs; }
    auto operator()(const Time& lhs, const StampedVariable& rhs) const -> bool { return lhs < rhs.stamp(); }
  };

  using StampedVariables = std::set<StampedVariable, StampedVariableCompare>;

  /// Evaluates the range.
  /// \return Range.
  [[nodiscard]] virtual auto range() const -> Range = 0;

  /// Elements accessor.
  /// \return Elements.
  auto elements() const -> const StampedVariables& { return stamped_variables_; }

  /// Elements modifier.
  /// \return Elements.
  auto elements() -> StampedVariables& { return stamped_variables_; }

  /// Variable pointers accessor.
  /// \return Pointers to (stamped) variables.
  [[nodiscard]] virtual auto variables() const -> std::vector<StampedVariable*> = 0;

  /// Time-based variable pointers accessor.
  /// \return Time-based pointers to (stamped) variables.
  [[nodiscard]] virtual auto variables(const Time& time) const -> std::vector<StampedVariable*> = 0;

  /// Parameter blocks accessor.
  /// \return Pointers to parameter blocks.
  [[nodiscard]] virtual auto parameterBlocks() const -> std::vector<Scalar*> = 0;

  /// Time-based parameter blocks accessor.
  /// \return Time-based pointers to parameter blocks.
  [[nodiscard]] virtual auto parameterBlocks(const Time& time) const -> std::vector<Scalar*> = 0;

  /// Evaluates this.
  /// \param time Query time.
  /// \param derivative Query derivative.
  /// \param jacobians Jacobians evaluation flag.
  /// \return Result.
  virtual auto evaluate(const Time& time, const Index& derivative, bool jacobians) const -> Result<TOutput> = 0;

  /// Evaluates this.
  /// \param time Query time.
  /// \param derivative Query derivative.
  /// \param jacobians Jacobians evaluation flag.
  /// \param elements Element pointers.
  /// \return Result.
  virtual auto evaluate(const Time& time, const Index& derivative, bool jacobians, const Scalar* const* elements) const -> Result<TOutput> = 0;

 protected:
  StampedVariables stamped_variables_;  ///< Stamped variables.
};

}  // namespace hyper::state
