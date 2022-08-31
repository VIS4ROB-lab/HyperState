/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/policies/forward.hpp"

#include "hyper/state/interpolators/polynomial.hpp"

namespace hyper {

/// Basis spline (i.e. B-Spline) interpolator for
/// uniform and non-uniform separation between bases and
/// arbitrary representation degree/order. We recommend using
/// odd degree splines due to symmetry.
class BasisInterpolator final : public PolynomialInterpolator {
 public:
  /// Evaluates the (uniform) mixing matrix.
  /// \return Interpolation matrix.
  [[nodiscard]] static auto Mixing(Order order) -> Matrix;

  /// Constructor from degree and uniformity flag.
  /// \param degree Input degree.
  /// \param uniform Input flag.
  explicit BasisInterpolator(Degree degree = 3, bool uniform = true);

  /// Updates the interpolator degree.
  /// \param degree Input degree.
  auto setDegree(Degree degree) -> void final;

  /// Retrieves the state layout.
  /// \return State layout.
  [[nodiscard]] auto layout() const -> StateLayout final;

  /// Evaluates the (non-uniform) mixing matrix.
  /// \param stamps Input stamps.
  /// \return Interpolation matrix.
  [[nodiscard]] auto mixing(const Stamps& stamps) const -> Matrix final;
};

} // namespace hyper
