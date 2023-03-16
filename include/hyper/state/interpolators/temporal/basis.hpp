/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/interpolators/temporal/polynomial.hpp"

namespace hyper::state {

/// Basis spline (i.e. B-Spline) interpolator for
/// uniform and non-uniform separation between bases and
/// arbitrary representation degree/order. We recommend using
/// odd degree splines due to symmetry.
template <typename TScalar, int TOrder>
class BasisInterpolator final : public PolynomialInterpolator<TScalar, TOrder> {
 public:
  // Definitions.
  using Base = PolynomialInterpolator<TScalar, TOrder>;

  using OrderVector = typename Base::OrderVector;
  using OrderMatrix = typename Base::OrderMatrix;

  /// Default constructor.
  explicit BasisInterpolator(int order = TOrder);

  /// Order setter.
  /// \param order Order.
  auto setOrder(int order) -> void final;

  /// Retrieves the layout.
  /// \param uniform Uniformity flag.
  /// \return Layout.
  [[nodiscard]] auto layout(bool uniform) const -> TemporalInterpolatorLayout final;

  /// Evaluates the (uniform) mixing matrix.
  /// \return Mixing matrix.
  [[nodiscard]] static auto Mixing(int order) -> OrderMatrix;

  /// Evaluates the (non-uniform) mixing matrix.
  /// \param inputs Input pointers (to stamped inputs).
  /// \param idx Time index into inputs.
  /// \return Interpolation matrix.
  [[nodiscard]] auto mixing(const TScalar* const* inputs, int idx) const -> OrderMatrix final;
};

}  // namespace hyper::state
