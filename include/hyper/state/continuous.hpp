/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <memory>
#include <set>
// #include <boost/container/flat_set.hpp>
// #include <absl/container/btree_set.h>

#include "hyper/state/interpolators/interpolators.hpp"
#include "hyper/state/temporal.hpp"
#include "hyper/variables/stamped.hpp"

namespace hyper::state {

template <typename TOutput, typename TVariable>
class ContinuousState : public TemporalState<TOutput, TVariable> {
 public:
  // Definitions.
  using Base = TemporalState<TOutput, TVariable>;

  using Index = typename Base::Index;
  using Scalar = typename Base::Scalar;

  using Time = typename Base::Time;
  using Range = typename Base::Range;

  using Variable = typename Base::Variable;
  using VariableTangent = typename Base::VariableTangent;
  using StampedVariable = typename Base::StampedVariable;
  using StampedVariableTangent = typename Base::StampedVariableTangent;

  using Output = typename Base::Output;
  using OutputTangent = typename Base::OutputTangent;
  using StampedOutput = typename Base::StampedOutput;
  using StampedOutputTangent = typename Base::StampedOutputTangent;

  using StampedVariables = typename Base::StampedVariables;

  /// Default constructor.
  ContinuousState();

  /// Constructor from interpolator.
  /// \param interpolator Interpolator.
  explicit ContinuousState(const TemporalInterpolator<Scalar>* interpolator);

  /// Evaluates the range.
  /// \return Range.
  [[nodiscard]] auto range() const -> Range final;

  /// Variable pointers accessor.
  /// \return Pointers to (stamped) variables.
  [[nodiscard]] auto variables() const -> std::vector<StampedVariable*> final;

  /// Time-based variable pointers accessor.
  /// \return Time-based pointers to (stamped) variables.
  [[nodiscard]] auto variables(const Time& time) const -> std::vector<StampedVariable*> final;

  /// Parameter blocks accessor.
  /// \return Pointers to parameter blocks.
  [[nodiscard]] auto parameterBlocks() const -> std::vector<Scalar*> final;

  /// Time-based parameter blocks accessor.
  /// \return Time-based pointers to parameter blocks.
  [[nodiscard]] auto parameterBlocks(const Time& time) const -> std::vector<Scalar*> final;

  /// Interpolator accessor.
  /// \return Interpolator.
  [[nodiscard]] auto interpolator() const -> const TemporalInterpolator<Scalar>*;

  /// Interpolator setter.
  /// \param interpolator Interpolator.
  auto setInterpolator(const TemporalInterpolator<Scalar>*) -> void;

  /// Retrieves the interpolator layout.
  /// \return Layout.
  auto layout() const -> TemporalInterpolator<Scalar>::Layout;

  /// Evaluates this.
  /// \param time Query time.
  /// \param derivative Query derivative.
  /// \param jacobian_type Requested type of Jacobian.
  /// \param stamped_variables Stamped variable pointers.
  /// \return Result.
  auto evaluate(const Time& time, const Index& derivative, JacobianType jacobian_type = JacobianType::NONE,  // NOLINT
                const Scalar* const* stamped_variables = nullptr) const -> Result<TOutput> final;

 private:
  // Definitions.
  using Iterator = typename StampedVariables::const_iterator;

  /// Retrieves the iterators for a time.
  /// \param time Query time.
  /// \return Iterators and number of elements between them.
  auto iterators(const Time& time) const -> std::tuple<Iterator, Iterator, Index>;

  const TemporalInterpolator<Scalar>* interpolator_;  ///< Interpolator.
};

}  // namespace hyper::state
