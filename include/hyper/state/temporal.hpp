/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <memory>
#include <set>

#include <glog/logging.h>

#include "hyper/state/forward.hpp"

#include "hyper/state/range.hpp"
#include "hyper/variables/jacobian.hpp"
#include "hyper/variables/stamped.hpp"
#include "hyper/variables/variable.hpp"

namespace hyper::state {

template <typename TOutput, typename TVariable>
class TemporalState {
 public:
  // Definitions.
  using Time = typename TOutput::Scalar;
  using Scalar = typename TOutput::Scalar;
  using Range = state::Range<Time, BoundaryPolicy::INCLUSIVE>;

  using Variable = TVariable;
  using VariableTangent = variables::Tangent<TVariable>;
  using StampedVariable = variables::Stamped<TVariable>;
  using StampedVariableTangent = variables::Stamped<VariableTangent>;

  using Output = TOutput;
  using OutputTangent = variables::Tangent<Output>;

  // Stamped variable compare.
  struct StampedVariableCompare {
    using is_transparent = std::true_type;
    auto operator()(const StampedVariable& lhs, const StampedVariable& rhs) const -> bool { return lhs.time() < rhs.time(); }
    auto operator()(const StampedVariable& lhs, const Time& rhs) const -> bool { return lhs.time() < rhs; }
    auto operator()(const Time& lhs, const StampedVariable& rhs) const -> bool { return lhs < rhs.time(); }
  };

  using StampedVariables = std::set<StampedVariable, StampedVariableCompare>;

  /// Constructor from uniformity flag and Jacobian type.
  /// \param is_uniform Uniformity flag.
  /// \param jacobian_type Jacobian type.
  explicit TemporalState(bool is_uniform = true, JacobianType jacobian_type = JacobianType::TANGENT_TO_STAMPED_MANIFOLD)
      : is_uniform_{is_uniform}, jacobian_type_{jacobian_type}, stamped_variables_{} {}

  /// Destructor.
  virtual ~TemporalState() = default;

  /// Jacobian type accessor.
  /// \return Jacobian type.
  [[nodiscard]] auto jacobianType() const -> JacobianType { return jacobian_type_; }

  /// Jacobian type setter.
  /// \param jacobian_type Jacobian type.
  inline auto setJacobianType(JacobianType jacobian_type) -> void { this->jacobian_type_ = jacobian_type; }

  /// Retrieves the ambient input size.
  /// \return Ambient input size.
  [[nodiscard]] constexpr auto ambientInputSize() const { return StampedVariable::kNumParameters; }

  /// Retrieves the ambient output size.
  /// \return Ambient output size.
  [[nodiscard]] constexpr auto ambientOutputSize() const { return Output::kNumParameters; }

  /// Retrieves the local input size.
  /// \return Local input size.
  [[nodiscard]] inline auto localInputSize() const {
    switch (this->jacobian_type_) {
      case JacobianType::TANGENT_TO_TANGENT:
        return VariableTangent::kNumParameters;
      case JacobianType::TANGENT_TO_STAMPED_TANGENT:
        return StampedVariableTangent::kNumParameters;
      case JacobianType::TANGENT_TO_MANIFOLD:
        return Variable::kNumParameters;
      case JacobianType::TANGENT_TO_STAMPED_MANIFOLD:
        return StampedVariable::kNumParameters;
      default: {
        LOG(FATAL) << "Unknown Jacobian type.";
        return -1;
      }
    }
  }

  /// Retrieves the local output size.
  /// \return Local output size.
  [[nodiscard]] constexpr auto localOutputSize() const { return OutputTangent::kNumParameters; }

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
  /// \param time Time.
  /// \param derivative Derivative.
  /// \param jacobian Flag.
  /// \param stamped_variables External pointers.
  /// \return Result.
  virtual auto evaluate(const Time& time, int derivative, bool jacobian = false, const Scalar* const* stamped_variables = nullptr) const -> Result<TOutput> = 0;  // NOLINT

 protected:
  bool is_uniform_;                     ///< Flag.
  JacobianType jacobian_type_;          ///< Jacobian type.
  StampedVariables stamped_variables_;  ///< Stamped variables.
};

}  // namespace hyper::state
