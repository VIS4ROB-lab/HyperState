/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/interpolators/interpolators.hpp"
#include "hyper/state/temporal.hpp"

namespace hyper::state {

template <typename TGroup>
class ContinuousState : public TemporalState<TGroup> {
 public:
  /// Constructor from interpolator, uniformity flag and Jacobian type.
  /// \param uniform Uniform.
  /// \param jacobian Jacobian.
  /// \param interpolator Interpolator.
  explicit ContinuousState(std::unique_ptr<TemporalInterpolator>&& interpolator, bool uniform = true, Jacobian jacobian = Jacobian::TANGENT_TO_STAMPED_GROUP);

  /// Updates the flag.
  /// \param flag Flag.
  auto setUniform(bool flag) -> void final;

  /// Evaluates the range.
  /// \return Range.
  [[nodiscard]] auto range() const -> InclusiveRange<Scalar> final;

  /// Time-based partition accessor.
  /// \param time Query time.
  /// \return Time-based partition.
  [[nodiscard]] auto partition(const Time& time) const -> Partition<Scalar*> final;

  /// Time-based parameter blocks accessor.
  /// \param time Query time.
  /// \return Time-based parameter blocks.
  [[nodiscard]] auto parameterBlocks(const Time& time) const -> std::vector<Scalar*> final;

  /// Interpolator accessor.
  /// \return Interpolator.
  [[nodiscard]] auto interpolator() const -> const TemporalInterpolator&;

  /// Interpolator setter.
  /// \param interpolator Interpolator.
  auto swapInterpolator(std::unique_ptr<TemporalInterpolator>& interpolator) -> void;

  /// Retrieves the interpolator layout.
  /// \return Layout.
  [[nodiscard]] auto layout() const -> const TemporalInterpolatorLayout&;

  /// Evaluates this.
  /// \param time Time.
  /// \param derivative Derivative.
  /// \param jacobian Flag.
  /// \param stamped_parameters External pointers.
  /// \return Result.
  auto evaluate(const Time& time, int derivative, bool jacobian = false, const Scalar* const* stamped_parameters = nullptr) const -> Result<TGroup> final;  // NOLINT

 private:
  // Definitions.
  using Iterator = typename TemporalState<TGroup>::StampedParameters::const_iterator;

  /// Retrieves the iterators for a time.
  /// \param time Query time.
  /// \return Iterators and number of parameters between them.
  auto iterators(const Time& time) const -> std::tuple<Iterator, Iterator, int>;

  TemporalInterpolatorLayout layout_;                   ///< Layout.
  std::unique_ptr<TemporalInterpolator> interpolator_;  ///< Interpolator.
};

}  // namespace hyper::state
