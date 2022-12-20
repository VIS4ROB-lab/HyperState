/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/motion/interpolators/temporal/forward.hpp"

namespace hyper {

template <typename TScalar>
class TemporalInterpolator {
 public:
  // Definitions.
  using Index = Eigen::Index;

  using Scalar = TScalar;
  using Layout = TemporalInterpolatorLayout<Index>;
  using Weights = MatrixX<Scalar>;

  /// Default destructor.
  virtual ~TemporalInterpolator() = default;

  /// Retrieves the layout.
  /// \return Layout.
  [[nodiscard]] virtual auto layout() const -> Layout = 0;

  /// Evaluates this.
  /// \param time Query time.
  /// \param derivative Query derivative.
  /// \param offset Offset to (left) central timestamp.
  /// \param timestamps Adjacent timestamps.
  /// \return Weights.
  virtual auto evaluate(const Scalar& time, const MotionDerivative& derivative, const Index& offset, const std::vector<Scalar>& timestamps) const -> Weights = 0;
};

} // namespace hyper
