/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <memory>
#include <set>
// #include <boost/container/flat_set.hpp>
// #include <absl/container/btree_set.h>

#include "hyper/motion/interpolators/spatial/spatial.hpp"
#include "hyper/motion/interpolators/temporal/temporal.hpp"
#include "hyper/motion/temporal.hpp"
#include "hyper/variables/stamped.hpp"

namespace hyper {

template <typename TVariable>
class ContinuousMotion : public TemporalMotion<TVariable> {
 public:
  // Definitions.
  using Base = TemporalMotion<TVariable>;
  using Scalar = typename Base::Scalar;

  using Time = typename Base::Time;
  using Range = typename Base::Range;

  using Element = typename Base::Element;

  /// Constructor from interpolator and policy.
  /// \param interpolator Input interpolator.
  explicit ContinuousMotion(std::unique_ptr<TemporalInterpolator<Scalar>>&& interpolator = nullptr);

  /// Evaluates the range.
  /// \return Range.
  [[nodiscard]] auto range() const -> Range final;

  /// Time-based pointers accessor.
  /// \return Time-based pointers.
  [[nodiscard]] virtual auto pointers(const Time& time) const -> Pointers<Element> final;

  /// Interpolator accessor.
  /// \return Interpolator.
  [[nodiscard]] auto interpolator() const -> const std::unique_ptr<TemporalInterpolator<Scalar>>&;

  /// Interpolator modifier.
  /// \return Interpolator.
  [[nodiscard]] auto interpolator() -> std::unique_ptr<TemporalInterpolator<Scalar>>&;

  /// Evaluates the motion.
  /// \param query Temporal motion query.
  /// \return True on success.
  auto evaluate(const TemporalMotionQuery<Scalar>& query) const -> bool final;

  /// Evaluates the motion.
  /// \param query Temporal motion query.
  /// \param pointers Input pointers.
  /// \return True on success.
  auto evaluate(const TemporalMotionQuery<Scalar>& query, const Scalar* const* pointers) const -> bool final;

  /// Evaluates the states.
  /// \param state_query State query.
  /// \return Interpolation result.
  auto evaluate(const StateQuery& state_query) const -> bool;

  /// Evaluates the states (with external parameters).
  /// \param state_query State query.
  /// \param raw_values Input values.
  /// \return Interpolation result.
  auto evaluate(const StateQuery& state_query, const Scalar* const* raw_values) const -> bool;

 private:
  std::unique_ptr<TemporalInterpolator<Scalar>> interpolator_; ///< Interpolator.
};

} // namespace hyper
