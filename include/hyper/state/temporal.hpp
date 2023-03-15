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

  using Scalar = typename Base::Scalar;

  using Time = Scalar;
  using Range = state::Range<Time, BoundaryPolicy::INCLUSIVE>;

  using Variable = TVariable;
  using VariableTangent = variables::Tangent<TVariable>;
  using StampedVariable = variables::Stamped<TVariable>;
  using StampedVariableTangent = variables::Stamped<VariableTangent>;

  using Output = TOutput;
  using OutputTangent = variables::Tangent<Output>;
  using StampedOutput = variables::Stamped<Output>;
  using StampedOutputTangent = variables::Stamped<OutputTangent>;

  // Stamped variable compare.
  struct StampedVariableCompare {
    using is_transparent = std::true_type;
    auto operator()(const StampedVariable& lhs, const StampedVariable& rhs) const -> bool { return lhs.time() < rhs.time(); }
    auto operator()(const StampedVariable& lhs, const Time& rhs) const -> bool { return lhs.time() < rhs; }
    auto operator()(const Time& lhs, const StampedVariable& rhs) const -> bool { return lhs < rhs.time(); }
  };

  using StampedVariables = std::set<StampedVariable, StampedVariableCompare>;

  /// Flag accessor.
  /// \return Flag.
  [[nodiscard]] inline auto isUniform() const -> bool { return is_uniform_; }

  /// Updates the flag.
  /// \param flag Flag.
  virtual inline auto setUniform(bool flag) -> void { is_uniform_ = flag; }

  /// Evaluates the range.
  /// \return Range.
  [[nodiscard]] virtual auto range() const -> Range = 0;

  /// Elements accessor.
  /// \return Elements.
  inline auto elements() const -> const StampedVariables& { return stamped_variables_; }

  /// Elements modifier.
  /// \return Elements.
  inline auto elements() -> StampedVariables& { return stamped_variables_; }

  /// Variable pointers accessor.
  /// \return Pointers to (stamped) variables.
  [[nodiscard]] virtual auto variables() const -> std::vector<variables::Variable<Scalar>*> = 0;

  /// Time-based variable pointers accessor.
  /// \return Time-based pointers to (stamped) variables.
  [[nodiscard]] virtual auto variables(const Time& time) const -> std::vector<variables::Variable<Scalar>*> = 0;

  /// Parameter blocks accessor.
  /// \return Pointers to parameter blocks.
  [[nodiscard]] virtual auto parameterBlocks() const -> std::vector<Scalar*> = 0;

  /// Time-based parameter blocks accessor.
  /// \return Time-based pointers to parameter blocks.
  [[nodiscard]] virtual auto parameterBlocks(const Time& time) const -> std::vector<Scalar*> = 0;

  /// Evaluates this.
  /// \param time Query time.
  /// \param derivative Query derivative.
  /// \param jacobian_type Requested type of Jacobian.
  /// \param stamped_variables Stamped variable pointers.
  /// \return Result.
  virtual auto evaluate(const Time& time, const Index& derivative, JacobianType jacobian_type = JacobianType::NONE,  // NOLINT
                        const Scalar* const* stamped_variables = nullptr) const -> Result<TOutput> = 0;

 protected:
  bool is_uniform_{true};               ///< Uniformity flag.
  StampedVariables stamped_variables_;  ///< Stamped variables.
};

}  // namespace hyper::state
