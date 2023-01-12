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
  using Elements = typename Base::Elements;

  /// Default constructor.
  ContinuousMotion();

  /// Constructor from interpolator.
  /// \param interpolator Interpolator.
  explicit ContinuousMotion(const TemporalInterpolator<Scalar>* interpolator);

  /// Evaluates the range.
  /// \return Range.
  [[nodiscard]] auto range() const -> Range final;

  /// Time-based pointers accessor.
  /// \return Time-based pointers.
  [[nodiscard]] virtual auto pointers(const Time& time) const -> Pointers<Element> final;

  /// Interpolator accessor.
  /// \return Interpolator.
  [[nodiscard]] auto interpolator() const -> const TemporalInterpolator<Scalar>*;

  /// Interpolator setter.
  /// \param interpolator Interpolator.
  auto setInterpolator(const TemporalInterpolator<Scalar>*) -> void;

  /// Evaluates this.
  /// \param time Query time.
  /// \param derivative Query derivative.
  /// \param jacobians Jacobians evaluation flag.
  /// \return Result.
  auto evaluate(const Time& time, const Index& derivative, bool jacobians) const -> TemporalMotionResult<TVariable> final;

  /// Evaluates this.
  /// \param time Query time.
  /// \param derivative Query derivative.
  /// \param jacobians Jacobians evaluation flag.
  /// \param elements Element pointers.
  /// \return Result.
  auto evaluate(const Time& time, const Index& derivative, bool jacobians, const Scalar* const* elements) const -> TemporalMotionResult<TVariable> final;

 private:
  // Definitions.
  using Iterator = typename Elements::const_iterator;

  /// Retrieves the iterators for a time.
  /// \param time Query time.
  /// \return Iterators and number of elements between them.
  auto iterators(const Time& time) const -> std::tuple<Iterator, Iterator, Index>;

  const TemporalInterpolator<Scalar>* interpolator_; ///< Interpolator.
};

} // namespace hyper
