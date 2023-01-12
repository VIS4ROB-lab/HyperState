/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/motion/interpolators/temporal/polynomial.hpp"

namespace hyper {

/// Basis spline (i.e. B-Spline) interpolator for
/// uniform and non-uniform separation between bases and
/// arbitrary representation degree/order. We recommend using
/// odd degree splines due to symmetry.
template <typename TScalar, int TOrder>
class BasisInterpolator final : public PolynomialInterpolator<TScalar, TOrder> {
 public:
  // Definitions.
  using Base = PolynomialInterpolator<TScalar, TOrder>;

  using Scalar = typename Base::Scalar;
  using Layout = typename Base::Layout;

  using OrderVector = typename Base::OrderVector;
  using OrderMatrix = typename Base::OrderMatrix;

  /// Default constructor.
  BasisInterpolator();

  /// Order setter.
  /// \param order Order.
  auto setOrder(const Index& order) -> void final;

  /// Retrieves the layout.
  /// \return Layout.
  [[nodiscard]] auto layout() const -> Layout final;

  /// Evaluates the (uniform) mixing matrix.
  /// \return Mixing matrix.
  [[nodiscard]] static auto Mixing(const Index& order) -> OrderMatrix;

  /// Evaluates the (non-uniform) mixing matrix.
  /// \param Times Input times.
  /// \return Interpolation matrix.
  [[nodiscard]] auto mixing(const std::vector<Scalar>& times) const -> OrderMatrix final;
};

} // namespace hyper
