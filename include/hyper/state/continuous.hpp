/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include <memory>
#include <set>
// #include <boost/container/flat_set.hpp>
// #include <absl/container/btree_set.h>

#include "hyper/state/interpolators/spatial/spatial.hpp"
#include "hyper/state/interpolators/temporal/temporal.hpp"
#include "hyper/state/temporal.hpp"
#include "hyper/variables/stamped.hpp"

namespace hyper::state {

template <typename TVariable>
class ContinuousState : public TemporalState<TVariable> {
 public:
  // Definitions.
  using Base = TemporalState<TVariable>;

  using Index = typename Base::Index;
  using Scalar = typename Base::Scalar;

  using Time = typename Base::Time;
  using Range = typename Base::Range;

  using Element = typename Base::Element;
  using Elements = typename Base::Elements;

  /// Default constructor.
  ContinuousState();

  /// Constructor from interpolator.
  /// \param interpolator Interpolator.
  explicit ContinuousState(const TemporalInterpolator<Scalar>* interpolator);

  /// Evaluates the range.
  /// \return Range.
  [[nodiscard]] auto range() const -> Range final;

  /// Element pointers accessor.
  /// \return Pointers to elements.
  [[nodiscard]] auto pointers() const -> std::vector<Element*> final;

  /// Time-based element pointers accessor.
  /// \return Time-based pointers to elements.
  [[nodiscard]] auto pointers(const Time& time) const -> std::vector<Element*> final;

  /// Element parameters accessor.
  /// \return Pointers to parameters.
  [[nodiscard]] auto parameters() const -> std::vector<Scalar*> final;

  /// Time-based parameters pointers accessor.
  /// \return Time-based pointers to parameters.
  [[nodiscard]] auto parameters(const Time& time) const -> std::vector<Scalar*> final;

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
  auto evaluate(const Time& time, const Index& derivative, bool jacobians) const -> Result<TVariable> final;

  /// Evaluates this.
  /// \param time Query time.
  /// \param derivative Query derivative.
  /// \param jacobians Jacobians evaluation flag.
  /// \param elements Element pointers.
  /// \return Result.
  auto evaluate(const Time& time, const Index& derivative, bool jacobians, const Scalar* const* elements) const -> Result<TVariable> final;

 private:
  // Definitions.
  using Iterator = typename Elements::const_iterator;

  /// Retrieves the iterators for a time.
  /// \param time Query time.
  /// \return Iterators and number of elements between them.
  auto iterators(const Time& time) const -> std::tuple<Iterator, Iterator, Index>;

  const TemporalInterpolator<Scalar>* interpolator_;  ///< Interpolator.
};

}  // namespace hyper::state
