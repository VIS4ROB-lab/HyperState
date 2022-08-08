/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <glog/logging.h>

#include "hyper/state/interpolators/basis.hpp"
#include "hyper/state/policies/abstract.hpp"

namespace hyper {

namespace {

using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

/// Evaluates low-level coefficients.
/// \param k Interpolation order.
/// \return Coefficients.
auto equidistantCoefficients(const Index k) -> std::pair<Matrix, Matrix> {
  Matrix Ak = Matrix::Zero(k, k - 1);
  Matrix Bk = Matrix::Zero(k, k - 1);

  for (Index i = 0; i < k - 1; ++i) {
    const auto a_i = Scalar{1} / static_cast<Scalar>(k - 1);
    Ak(i, i) = static_cast<Scalar>(i + 1) * a_i;
    Ak(i + 1, i) = static_cast<Scalar>(k - 2 - i) * a_i;
    Bk(i, i) = -a_i;
    Bk(i + 1, i) = a_i;
  }

  return {Ak, Bk};
}

/// Evaluates the matrix precursor (i.e. non-cumulative matrix).
/// \param k Interpolation order.
/// \return Matrix precursor.
auto equidistantRecursion(const Index k) -> Matrix { // NOLINT
  if (k == 1) {
    Matrix matrix = Matrix::Ones(1, 1);
    return matrix;
  } else if (k == 2) {
    Matrix matrix;
    matrix.resize(k, k);
    matrix(0, 0) = Scalar{1};
    matrix(1, 0) = Scalar{0};
    matrix(0, 1) = Scalar{-1};
    matrix(1, 1) = Scalar{1};
    return matrix;
  } else {
    Matrix AkMk, BkMk;
    AkMk.resize(k, k);
    BkMk.resize(k, k);
    const auto [Ak, Bk] = equidistantCoefficients(k);
    const auto Mk = equidistantRecursion(k - 1);
    AkMk.topLeftCorner(k, k - 1) = Ak * Mk;
    AkMk.col(k - 1).setZero();
    BkMk.col(0).setZero();
    BkMk.topRightCorner(k, k - 1) = Bk * Mk;
    return AkMk + BkMk;
  }
}

/// Evaluates low-level coefficients.
/// \param stamps Input stamps (i.e. times associated with variables).
/// \param i Index.
/// \param j Index.
/// \return Coefficients.
auto coefficient(const Stamps& stamps, const Index i, const Index j) -> std::pair<Scalar, Scalar> {
  const auto tj0 = stamps[j - 2];
  const auto tj1 = stamps[j - 1];
  const auto ti = stamps[i];
  const auto tij = stamps[i + j - 1];
  const auto inverse = Scalar{1} / (tij - ti);
  return {(tj0 - ti) * inverse, (tj1 - tj0) * inverse};
}

/// Evaluates low-level coefficient matrices.
/// \param stamps Input stamps (i.e. times associated with variables).
/// \param k Index.
/// \return Coefficient matrices.
auto coefficients(const Stamps& stamps, const Index k) -> std::pair<Matrix, Matrix> {
  Matrix Ak = Matrix::Zero(k, k - 1);
  Matrix Bk = Matrix::Zero(k, k - 1);

  for (Index i = 0; i < k - 1; ++i) {
    const auto [a, b] = coefficient(stamps, i, k);
    Ak(i, i) = Scalar{1} - a;
    Ak(i + 1, i) = a;
    Bk(i, i) = -b;
    Bk(i + 1, i) = b;
  }

  return {Ak, Bk};
}

/// Evaluates the matrix precursor (i.e. non-cumulative matrix).
/// \param stamps Input stamps (i.e. times associated with variables).
/// \param k Index.
/// \return Matrix precursor.
auto recursion(const Stamps& stamps, const Index k) -> Matrix { // NOLINT
  if (k == 1) {
    Matrix matrix = Matrix::Ones(1, 1);
    return matrix;
  } else if (k == 2) {
    Matrix matrix;
    matrix.resize(k, k);
    matrix(0, 0) = Scalar{1};
    matrix(1, 0) = Scalar{0};
    matrix(0, 1) = Scalar{-1};
    matrix(1, 1) = Scalar{1};
    return matrix;
  } else { // Recursion.
    Matrix AkMk, BkMk;
    AkMk.resize(k, k);
    BkMk.resize(k, k);
    const auto [Ak, Bk] = coefficients(stamps, k);
    const auto Mk = recursion(stamps, k - 1);
    AkMk.topLeftCorner(k, k - 1) = Ak * Mk;
    AkMk.col(k - 1).setZero();
    BkMk.col(0).setZero();
    BkMk.topRightCorner(k, k - 1) = Bk * Mk;
    return AkMk + BkMk;
  }
}

} // namespace

auto BasisInterpolator::Mixing(const Order order) -> Matrix {
  return Matrix::Ones(order, order).triangularView<Eigen::Upper>() * equidistantRecursion(order);
}

BasisInterpolator::BasisInterpolator(const Degree degree, const bool uniform)
    : PolynomialInterpolator{uniform} {
  setDegree(degree);
}

auto BasisInterpolator::setDegree(const Degree degree) -> void {
  degree_ = degree;
  order_ = degree + 1;
  mixing_ = Mixing(order_);
  polynomials_ = polynomials();
}

auto BasisInterpolator::layout() const -> InterpolatorLayout {
  if (uniform_) {
    const auto index = -(degree_ / 2);
    return {{index, index + order_}, {0, order_}};
  } else {
    const auto index = -(degree_ - 1);
    const auto offset = (degree_ - 1) / 2;
    return {{index, index + 2 * degree_}, {offset, offset + order_}};
  }
}

auto BasisInterpolator::mixing(const Stamps& stamps) const -> Matrix {
  return Matrix::Ones(order_, order_).triangularView<Eigen::Upper>() * recursion(stamps, order_);
}

} // namespace hyper
