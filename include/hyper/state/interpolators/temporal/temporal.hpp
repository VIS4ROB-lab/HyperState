/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/matrix.hpp"
#include "hyper/state/interpolators/temporal/forward.hpp"

namespace hyper::state {

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
  /// \param uniform Uniformity flag.
  /// \return Layout.
  [[nodiscard]] virtual auto layout(bool uniform) const -> Layout = 0;

  /// Evaluates this.
  /// \param ut Normalized time.
  /// \param i_dt Normalization delta.
  /// \param derivative Query derivative.
  /// \param timestamps Adjacent timestamps.
  /// \return Weights.
  virtual auto evaluate(const Scalar& ut, const Scalar& i_dt, const Index& derivative, const std::vector<Scalar>* timestamps) const -> Weights = 0;
};

}  // namespace hyper::state
