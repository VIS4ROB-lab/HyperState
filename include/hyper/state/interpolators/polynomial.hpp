/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/state/policies/forward.hpp"

#include "hyper/state/interpolators/abstract.hpp"

namespace hyper {

/// Basis spline (i.e. B-Spline) interpolator for
/// uniform and non-uniform separation between bases and
/// arbitrary representation degree/order. We recommend using
/// odd degree splines due to symmetry.
class PolynomialInterpolator : public AbstractInterpolator<Scalar> {
 public:
  // Definitions.
  using Degree = Index;
  using Order = Index;
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  /// Updates the uniformity flag.
  /// \param uniform Input flag.
  auto setUniform(bool uniform = true) -> void;

  /// Updates the interpolator degree.
  /// \param degree Input degree.
  virtual auto setDegree(Degree degree) -> void = 0;

  /// Evaluates the (non-uniform) mixing matrix.
  /// \param times Input times.
  /// \return Interpolation matrix.
  [[nodiscard]] virtual auto mixing(const Times& times) const -> Matrix = 0;

  /// Evaluates this.
  /// \param query Query.
  /// \return True on success.
  [[nodiscard]] auto evaluate(const Query& query) const -> bool final;

 protected:
  /// Constructor from uniformity flag.
  /// \param uniform Input flag.
  explicit PolynomialInterpolator(bool uniform);

  /// Computes polynomial coefficient matrix.
  /// \return Polynomial coefficient matrix.
  [[nodiscard]] auto polynomials() const -> Matrix;

  /// Polynomial derivatives of normalized stamp.
  /// \param time Normalized time.
  /// \param i Derivative order.
  /// \return Polynomial derivatives.
  [[nodiscard]] auto polynomial(const Time& time, Index i) const -> Matrix;

  bool uniform_;       ///< Uniformity flag.
  Degree degree_;      ///< Degree.
  Order order_;        ///< Order.
  Matrix mixing_;      ///< Cached mixing matrix.
  Matrix polynomials_; ///< Cached polynomial derivatives.
};

} // namespace hyper
