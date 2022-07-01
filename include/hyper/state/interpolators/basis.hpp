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
class BasisInterpolator final : public AbstractInterpolator {
 public:
  /// Evaluates the (uniform) mixing matrix.
  /// \return Interpolation matrix.
  [[nodiscard]] static auto Mixing(Index order) -> Matrix;

  /// Constructor from degree and uniformity flag.
  /// \param degree Input degree.
  /// \param uniform Input flag.
  explicit BasisInterpolator(Index degree = 3, bool uniform = true);

  /// Default destructor.
  ~BasisInterpolator() final = default;

  /// Updates the interpolator degree.
  /// \param degree Input degree.
  auto setDegree(Index degree) -> void;

  /// Updates the uniformity flag.
  /// \param uniform Input flag.
  auto setUniform(bool uniform = true) -> void;

  /// Retrieves the interpolator layout.
  /// \return Interpolator layout.
  [[nodiscard]] auto layout() const -> InterpolatorLayout final;

  /// Evaluates the mixing matrix.
  /// \param stamps Input stamps.
  /// \return Interpolation matrix.
  [[nodiscard]] auto mixing(const Stamps& stamps) const -> Matrix;

  /// Evaluates the weights.
  /// \param state_query State query.
  /// \param stamps Stamps of the variables.
  /// \return Weights.
  [[nodiscard]] auto weights(const StateQuery& state_query, const Stamps& stamps) const -> Matrix final;

 private:
  /// Computes polynomial coefficient matrix.
  /// \return Polynomial coefficient matrix.
  [[nodiscard]] auto polynomials() const -> Matrix;

  /// Polynomial derivatives of normalized stamp.
  /// \param stamp Normalized stamp.
  /// \param i Derivative order.
  /// \return Polynomial derivatives.
  [[nodiscard]] auto polynomial(const Stamp& stamp, Index i) const -> Matrix;

  bool uniform_;       ///< Uniformity flag.
  Index degree_;       ///< Degree.
  Index order_;        ///< Order.
  Matrix polynomials_; ///< Matrix of polynomial derivatives.
  Matrix mixing_;      ///< Cached mixing matrix.
};

} // namespace hyper
