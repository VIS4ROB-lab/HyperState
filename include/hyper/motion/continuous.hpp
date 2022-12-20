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
  using Query = typename Base::Query;
  using Result = typename Base::Result;

  /// Constructor from temporal interpolator.
  /// \param temporal_interpolator Temporal interpolator.
  explicit ContinuousMotion(std::unique_ptr<TemporalInterpolator<Scalar>>&& temporal_interpolator = nullptr);

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
  auto evaluate(const Query& query) const -> Result final;

  /// Evaluates the motion.
  /// \param query Temporal motion query.
  /// \param inputs Input pointers.
  /// \return True on success.
  auto evaluate(const Query& query, const Scalar* const* inputs) const -> Result final;

 private:
  // Definitions.
  using Iterator = typename Elements::const_iterator;

  /// Retrieves the iterators for a time.
  /// \param time Query time.
  /// \return Iterators and number of elements between them.
  auto iterators(const Time& time) const -> std::tuple<Iterator, Iterator, Index>;

  std::unique_ptr<TemporalInterpolator<Scalar>> temporal_interpolator_; ///< Temporal interpolator.
};

} // namespace hyper
