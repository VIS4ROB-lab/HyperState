/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/matrix.hpp"
#include "hyper/motion/interpolators/temporal/forward.hpp"

namespace hyper {

template <typename TScalar>
class TemporalInterpolator {
 public:
  // Definitions.
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
  /// \param timestamps Adjacent timestamps.
  /// \param offset Offset to (left) central timestamp.
  /// \return Weights.
  virtual auto evaluate(const Scalar& time, const Index& derivative, const std::vector<Scalar>& timestamps, const Index& offset) const -> Weights = 0;
};

} // namespace hyper
