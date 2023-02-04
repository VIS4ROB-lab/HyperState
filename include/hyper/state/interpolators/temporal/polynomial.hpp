/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/matrix.hpp"
#include "hyper/state/interpolators/temporal/temporal.hpp"
#include "hyper/vector.hpp"

namespace hyper::state {

template <typename TScalar, int TOrder>
class PolynomialInterpolator : public TemporalInterpolator<TScalar> {
 public:
  // Definitions.
  using Base = TemporalInterpolator<TScalar>;

  using Index = typename Base::Index;

  using Scalar = typename Base::Scalar;
  using Layout = typename Base::Layout;

  using OrderVector = Vector<Scalar, TOrder>;
  using OrderMatrix = Matrix<Scalar, TOrder, TOrder>;

  using Weights = typename Base::Weights;

  /// Computes polynomial coefficient matrix.
  /// \return Polynomial coefficient matrix.
  static auto Polynomials(const Index& order) -> OrderMatrix;

  /// Order accessor.
  /// \return Order.
  [[nodiscard]] auto order() const -> Index;

  /// Order setter.
  /// \param order Order.
  virtual auto setOrder(const Index& order) -> void = 0;

  /// Evaluates the (non-uniform) mixing matrix.
  /// \param inputs Input pointers (to stamped inputs).
  /// \param idx Time index into inputs.
  /// \return Interpolation matrix.
  [[nodiscard]] virtual auto mixing(const Scalar* const* inputs, const Index& idx) const -> OrderMatrix = 0;

  /// Evaluates this.
  /// \param ut Normalized time.
  /// \param i_dt Normalization delta.
  /// \param derivative Query derivative.
  /// \param inputs Input pointers (to stamped inputs).
  /// \param idx Time index into inputs.
  /// \return Weights.
  auto evaluate(const Scalar& ut, const Scalar& i_dt, const Index& derivative, const Scalar* const* inputs, const Index& idx) const -> Weights final;

 protected:
  OrderMatrix mixing_;       ///< Cached mixing matrix.
  OrderMatrix polynomials_;  ///< Cached polynomial derivatives.
};

}  // namespace hyper::state
