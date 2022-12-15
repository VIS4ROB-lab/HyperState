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
/// \param times Input times (i.e. times associated with variables).
/// \param i Index.
/// \param j Index.
/// \return Coefficients.
auto coefficient(const Times& times, const Index i, const Index j) -> std::pair<Scalar, Scalar> {
  const auto tj0 = times[j - 2];
  const auto tj1 = times[j - 1];
  const auto ti = times[i];
  const auto tij = times[i + j - 1];
  const auto inverse = Scalar{1} / (tij - ti);
  return {(tj0 - ti) * inverse, (tj1 - tj0) * inverse};
}

/// Evaluates low-level coefficient matrices.
/// \param times Input times (i.e. times associated with variables).
/// \param k Index.
/// \return Coefficient matrices.
auto coefficients(const Times& times, const Index k) -> std::pair<Matrix, Matrix> {
  Matrix Ak = Matrix::Zero(k, k - 1);
  Matrix Bk = Matrix::Zero(k, k - 1);

  for (Index i = 0; i < k - 1; ++i) {
    const auto [a, b] = coefficient(times, i, k);
    Ak(i, i) = Scalar{1} - a;
    Ak(i + 1, i) = a;
    Bk(i, i) = -b;
    Bk(i + 1, i) = b;
  }

  return {Ak, Bk};
}

/// Evaluates the matrix precursor (i.e. non-cumulative matrix).
/// \param times Input times (i.e. times associated with variables).
/// \param k Index.
/// \return Matrix precursor.
auto recursion(const Times& times, const Index k) -> Matrix { // NOLINT
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
    const auto [Ak, Bk] = coefficients(times, k);
    const auto Mk = recursion(times, k - 1);
    AkMk.topLeftCorner(k, k - 1) = Ak * Mk;
    AkMk.col(k - 1).setZero();
    BkMk.col(0).setZero();
    BkMk.topRightCorner(k, k - 1) = Bk * Mk;
    return AkMk + BkMk;
  }
}

} // namespace

template <typename TScalar, int TOrder>
BasisInterpolator<TScalar, TOrder>::BasisInterpolator() {
  if (0 < Base::kOrder) {
    setOrder(Base::kOrder);
  }
}

template <typename TScalar, int TOrder>
auto BasisInterpolator<TScalar, TOrder>::setOrder(const Index& order) -> void {
  if (Base::kOrder < 0) {
    this->mixing_ = Mixing(order);
    this->polynomials_ = this->polynomials();
  } else {
    DCHECK_EQ(order, Base::kOrder);
  }
}

template <typename TScalar, int TOrder>
auto BasisInterpolator<TScalar, TOrder>::layout() const -> Layout {
  const auto order = this->order();

  if (this->isUniform()) {
    if (order % 2 == 0) {
      const auto margin = order / 2;
      return {
          .outer_input_size = order,
          .inner_input_size = order,
          .left_input_margin = margin,
          .right_input_margin = margin,
          .left_input_padding = 0,
          .right_input_padding = 0,
          .output_size = order};
    } else {
      const auto margin = (order - 1) / 2;
      return {
          .outer_input_size = order,
          .inner_input_size = order,
          .left_input_margin = margin + 1,
          .right_input_margin = margin,
          .left_input_padding = 0,
          .right_input_padding = 0,
          .output_size = order};
    }
  } else {
    const auto degree = order - 1;
    const auto padding = (degree - 1) / 2;
    return {
        .outer_input_size = 2 * degree,
        .inner_input_size = order,
        .left_input_margin = degree,
        .right_input_margin = degree,
        .left_input_padding = padding,
        .right_input_padding = padding,
        .output_size = order};
  }
}

template <typename TScalar, int TOrder>
auto BasisInterpolator<TScalar, TOrder>::Mixing(const Index& order) -> OrderMatrix {
  return OrderMatrix::Ones(order, order).template triangularView<Eigen::Upper>() * equidistantRecursion(order);
}

template <typename TScalar, int TOrder>
auto BasisInterpolator<TScalar, TOrder>::mixing(const Times& times) const -> OrderMatrix {
  const auto order = this->order();
  return OrderMatrix::Ones(order, order).template triangularView<Eigen::Upper>() * recursion(times, order);
}

template class BasisInterpolator<double, Eigen::Dynamic>;

} // namespace hyper
