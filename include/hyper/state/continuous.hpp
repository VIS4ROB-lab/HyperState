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

template <typename TElement>
class ContinuousState : public TemporalState<TElement> {
 public:
  // Definitions.
  using Base = TemporalState<TElement>;

  using Scalar = typename Base::Scalar;
  using Time = typename Base::Time;
  using Range = typename Base::Range;

  using Element = typename Base::Element;
  using ElementTangent = typename Base::ElementTangent;
  using StampedElement = typename Base::StampedElement;
  using StampedElementTangent = typename Base::StampedElementTangent;
  using StampedElements = typename Base::StampedElements;

  /// Constructor from interpolator, uniformity flag and Jacobian type.
  /// \param is_uniform Uniformity flag.
  /// \param jacobian_type Jacobian type.
  /// \param interpolator Interpolator.
  explicit ContinuousState(std::unique_ptr<TemporalInterpolator<Scalar>>&& interpolator, bool is_uniform = true, JacobianType jacobian_type = Base::kDefaultJacobianType);

  /// Updates the flag.
  /// \param flag Flag.
  auto setUniform(bool flag) -> void final;

  /// Evaluates the range.
  /// \return Range.
  [[nodiscard]] auto range() const -> Range final;

  /// Time-based partition accessor.
  /// \param time Query time.
  /// \return Time-based partition.
  [[nodiscard]] auto partition(const Time& time) const -> variables::Partition<Scalar*> final;

  /// Time-based parameter blocks accessor.
  /// \param time Query time.
  /// \return Time-based parameter blocks.
  [[nodiscard]] auto parameterBlocks(const Time& time) const -> std::vector<Scalar*> final;

  /// Interpolator accessor.
  /// \return Interpolator.
  [[nodiscard]] auto interpolator() const -> const TemporalInterpolator<Scalar>&;

  /// Interpolator setter.
  /// \param interpolator Interpolator.
  auto swapInterpolator(std::unique_ptr<TemporalInterpolator<Scalar>>& interpolator) -> void;

  /// Retrieves the interpolator layout.
  /// \return Layout.
  [[nodiscard]] auto layout() const -> const TemporalInterpolatorLayout&;

  /// Evaluates this.
  /// \param time Time.
  /// \param derivative Derivative.
  /// \param jacobian Flag.
  /// \param stamped_elements External pointers.
  /// \return Result.
  auto evaluate(const Time& time, int derivative, bool jacobian = false, const Scalar* const* stamped_elements = nullptr) const -> Result<TElement> final;  // NOLINT

 private:
  // Definitions.
  using Iterator = typename StampedElements::const_iterator;

  /// Retrieves the iterators for a time.
  /// \param time Query time.
  /// \return Iterators and number of elements between them.
  auto iterators(const Time& time) const -> std::tuple<Iterator, Iterator, int>;

  TemporalInterpolatorLayout layout_;                           ///< Layout.
  std::unique_ptr<TemporalInterpolator<Scalar>> interpolator_;  ///< Interpolator.
};

}  // namespace hyper::state
