/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/matrix.hpp"
#include "hyper/state/interpolators/temporal/temporal.hpp"

namespace hyper::state {

template <int TOrder>
class PolynomialInterpolator : public TemporalInterpolator {
 public:
  // Definitions.
  using OrderVector = Vector<TOrder>;
  using OrderMatrix = Matrix<TOrder, TOrder>;

  /// Computes polynomial coefficient matrix.
  /// \return Polynomial coefficient matrix.
  static auto Polynomials(int order) -> OrderMatrix;

  /// Order accessor.
  /// \return Order.
  [[nodiscard]] auto order() const -> int;

  /// Order setter.
  /// \param order Order.
  virtual auto setOrder(int order) -> void = 0;

  /// Evaluates the (non-uniform) mixing matrix.
  /// \param inputs Input pointers (to stamped inputs).
  /// \param idx Time index into inputs.
  /// \return Interpolation matrix.
  [[nodiscard]] virtual auto mixing(const Scalar* const* inputs, int idx) const -> OrderMatrix = 0;

  /// Evaluates this.
  /// \param ut Normalized time.
  /// \param i_dt Normalization delta.
  /// \param derivative Query derivative.
  /// \param inputs Input pointers (to stamped inputs).
  /// \param idx Time index into inputs.
  /// \return Weights.
  auto evaluate(const Scalar& ut, const Scalar& i_dt, int derivative, const Scalar* const* inputs, int idx) const -> MatrixX final;

 protected:
  OrderMatrix mixing_;       ///< Cached mixing matrix.
  OrderMatrix polynomials_;  ///< Cached polynomial derivatives.
};

}  // namespace hyper::state
