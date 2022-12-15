/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/interpolators/forward.hpp"

namespace hyper {

/// Abstract interface for interpolation methods.
class AbstractInterpolator {
 public:
  // Definitions.
  using Layout = TemporalInterpolatorLayout<Eigen::Index>;
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  /// Default destructor.
  virtual ~AbstractInterpolator() = default;

  /// Retrieves the layout.
  /// \return Layout.
  [[nodiscard]] virtual auto layout() const -> Layout = 0;

  /// Evaluates the weights.
  /// \param time Query time.
  /// \param times Times of the variables.
  /// \param derivative Highest requested derivative.
  /// \return Weights.
  [[nodiscard]] virtual auto weights(const Time& time, const Times& times, Index derivative) const -> Matrix = 0;
};

} // namespace hyper
