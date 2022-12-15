/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/policies/forward.hpp"

#include "hyper/matrix.hpp"
#include "hyper/vector.hpp"
#include "temporal.hpp"

namespace hyper {

template <typename TScalar, int TOrder>
class PolynomialInterpolator : public TemporalInterpolator<TScalar> {
 public:
  // Definitions.
  using Base = TemporalInterpolator<TScalar>;

  using Index = typename Base::Index;
  using Scalar = typename Base::Scalar;
  using Layout = typename Base::Layout;
  using Query = typename Base::Query;

  using OrderVector = Vector<Scalar, TOrder>;
  using OrderMatrix = Matrix<Scalar, TOrder, TOrder>;

  using Weights = Matrix<Scalar, TOrder, Eigen::Dynamic>;

  /// Uniformity flag accessor.
  /// \return Uniformity flag.
  [[nodiscard]] auto isUniform() const -> bool;

  /// Set uniform.
  auto setUniform() -> void;

  /// Set non-uniform.
  auto setNonUniform() -> void;

  /// Order accessor.
  /// \return Order.
  [[nodiscard]] auto order() const -> Index;

  /// Order setter.
  /// \param order Order.
  virtual auto setOrder(const Index& order) -> void = 0;

  /// Evaluates the (non-uniform) mixing matrix.
  /// \param times Input times.
  /// \return Interpolation matrix.
  [[nodiscard]] virtual auto mixing(const Times& times) const -> OrderMatrix = 0;

  /// Evaluates this.
  /// \param query Query.
  /// \return True on success.
  [[nodiscard]] auto evaluate(const Query& query) const -> bool final;

 protected:
  /// Computes polynomial coefficient matrix.
  /// \return Polynomial coefficient matrix.
  [[nodiscard]] auto polynomials() const -> OrderMatrix;

  /// Polynomial derivatives of normalized stamp.
  /// \param ut Normalized time.
  /// \param i Derivative order.
  /// \return Polynomial derivatives.
  [[nodiscard]] auto polynomial(const Time& ut, const Index& i) const -> OrderVector;

  bool is_uniform_{true}; ///< Uniformity flag.

  OrderMatrix mixing_;      ///< Cached mixing matrix.
  OrderMatrix polynomials_; ///< Cached polynomial derivatives.
};

} // namespace hyper
