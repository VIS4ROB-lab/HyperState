/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/interpolators/forward.hpp"

namespace hyper {

/// Abstract interface for interpolation methods.
class AbstractInterpolator {
 public:
  // Definitions.
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  /// Default destructor.
  virtual ~AbstractInterpolator() = default;

  /// Retrieves the interpolator layout.
  /// \return Interpolator layout.
  [[nodiscard]] virtual auto layout() const -> InterpolatorLayout = 0;

  /// Evaluates the weights.
  /// \param stamp Query stamp.
  /// \param stamps Stamps of the variables.
  /// \param derivative Highest requested derivative.
  /// \return Weights.
  [[nodiscard]] virtual auto weights(const Stamp& stamp, const Stamps& stamps, Index derivative) const -> Matrix = 0;
};

} // namespace hyper
