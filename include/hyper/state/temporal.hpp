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

template <typename TOutput, typename TInput>
class TemporalState : public State<typename TInput::Scalar> {
 public:
  // Definitions.
  using Base = State<typename TInput::Scalar>;

  using Index = typename Base::Index;
  using Scalar = typename Base::Scalar;

  using Time = Scalar;
  using Range = state::Range<Time, BoundaryPolicy::INCLUSIVE>;

  using Input = TInput;
  using InputTangent = variables::Tangent<TInput>;
  using StampedInput = variables::Stamped<TInput>;
  using StampedInputTangent = variables::Tangent<StampedInput>;

  using Output = TOutput;
  using OutputTangent = variables::Tangent<Output>;
  using StampedOutput = variables::Stamped<Output>;
  using StampedOutputTangent = variables::Stamped<OutputTangent>;

  // Stamped input compare.
  struct StampedInputCompare {
    using is_transparent = std::true_type;
    auto operator()(const StampedInput& lhs, const StampedInput& rhs) const -> bool { return lhs.stamp() < rhs.stamp(); }
    auto operator()(const StampedInput& lhs, const Time& rhs) const -> bool { return lhs.stamp() < rhs; }
    auto operator()(const Time& lhs, const StampedInput& rhs) const -> bool { return lhs < rhs.stamp(); }
  };

  using StampedInputs = std::set<StampedInput, StampedInputCompare>;

  /// Evaluates the range.
  /// \return Range.
  [[nodiscard]] virtual auto range() const -> Range = 0;

  /// Elements accessor.
  /// \return Elements.
  auto elements() const -> const StampedInputs& { return stamped_inputs_; }

  /// Elements modifier.
  /// \return Elements.
  auto elements() -> StampedInputs& { return stamped_inputs_; }

  /// Input pointers accessor.
  /// \return Pointers to (stamped) inputs.
  [[nodiscard]] virtual auto inputs() const -> std::vector<StampedInput*> = 0;

  /// Time-based input pointers accessor.
  /// \return Time-based pointers to (stamped) inputs.
  [[nodiscard]] virtual auto inputs(const Time& time) const -> std::vector<StampedInput*> = 0;

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
  /// \param inputs Input pointers (to stamped inputs).
  /// \return Result.
  virtual auto evaluate(const Time& time, const Index& derivative, bool jacobians, const Scalar* const* inputs) const -> Result<TOutput> = 0;

 protected:
  StampedInputs stamped_inputs_;  ///< Stamped inputs.
};

}  // namespace hyper::state
