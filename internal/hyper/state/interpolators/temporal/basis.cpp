/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <glog/logging.h>

#include "hyper/state/interpolators/temporal/basis.hpp"

namespace hyper::state {

namespace {

/// Evaluates low-level coefficients.
template <typename TScalar, typename TIndex, int TRows, int TCols = TRows != Eigen::Dynamic ? (TRows - 1) : Eigen::Dynamic>
auto uniformCoefficients(const TIndex& k) -> std::pair<Matrix<TScalar, TRows, TCols>, Matrix<TScalar, TRows, TCols>> {
  using Matrix = Matrix<TScalar, TRows, TCols>;

  Matrix Ak = Matrix::Zero(k, k - 1);
  Matrix Bk = Matrix::Zero(k, k - 1);

  for (TIndex i = 0; i < k - 1; ++i) {
    const auto a_i = TScalar{1} / static_cast<TScalar>(k - 1);
    Ak(i, i) = static_cast<TScalar>(i + 1) * a_i;
    Ak(i + 1, i) = static_cast<TScalar>(k - 2 - i) * a_i;
    Bk(i, i) = -a_i;
    Bk(i + 1, i) = a_i;
  }

  return {Ak, Bk};
}

/// Evaluates the weight precursor (i.e. non-cumulative matrix).
template <typename TScalar, typename TIndex, int K>
auto uniformRecursion(const TIndex& k) -> Matrix<TScalar, K, K> {  // NOLINT
  using Matrix = Matrix<TScalar, K, K>;

  if (k == 1) {
    Matrix matrix = Matrix::Ones(1, 1);
    return matrix;
  } else if (k == 2) {
    Matrix matrix{k, k};
    matrix(0, 0) = TScalar{1};
    matrix(1, 0) = TScalar{0};
    matrix(0, 1) = TScalar{-1};
    matrix(1, 1) = TScalar{1};
    return matrix;
  } else {
    const auto [Ak, Bk] = uniformCoefficients<TScalar, TIndex, K>(k);
    const auto Mk = uniformRecursion<TScalar, TIndex, ((K > 1) ? (K - 1) : Eigen::Dynamic)>(k - 1);
    Matrix AkMk{k, k};
    Matrix BkMk{k, k};
    AkMk.topLeftCorner(k, k - 1) = Ak * Mk;
    AkMk.col(k - 1).setZero();
    BkMk.col(0).setZero();
    BkMk.topRightCorner(k, k - 1) = Bk * Mk;
    return AkMk + BkMk;
  }
}

/// Evaluates low-level coefficients.
template <typename TScalar, typename TIndex>
auto coefficient(const TIndex& i, const TIndex& j, const TScalar* const* inputs, const TIndex& idx) -> std::pair<TScalar, TScalar> {
  const auto v2 = inputs[j - 2][idx];
  const auto v1 = inputs[j - 1][idx];
  const auto vi = inputs[i][idx];
  const auto vij = inputs[i + j - 1][idx];
  const auto i_t = TScalar{1} / (vij - vi);
  return {(v2 - vi) * i_t, (v1 - v2) * i_t};
}

/// Evaluates low-level coefficient matrices.
template <typename TScalar, typename TIndex, int TRows, int TCols = TRows != Eigen::Dynamic ? (TRows - 1) : Eigen::Dynamic>
auto coefficients(const TIndex& k, const TScalar* const* inputs, const TIndex& idx) -> std::pair<Matrix<TScalar, TRows, TCols>, Matrix<TScalar, TRows, TCols>> {
  using Matrix = Matrix<TScalar, TRows, TCols>;

  Matrix Ak = Matrix::Zero(k, k - 1);
  Matrix Bk = Matrix::Zero(k, k - 1);

  for (TIndex i = 0; i < k - 1; ++i) {
    const auto [a, b] = coefficient<TScalar, TIndex>(i, k, inputs, idx);
    Ak(i, i) = TScalar{1} - a;
    Ak(i + 1, i) = a;
    Bk(i, i) = -b;
    Bk(i + 1, i) = b;
  }

  return {Ak, Bk};
}

/// Evaluates the weight precursor (i.e. non-cumulative matrix).
template <typename TScalar, typename TIndex, int K>
auto nonUniformRecursion(const TIndex& k, const TScalar* const* inputs, const TIndex& idx) -> Matrix<TScalar, K, K> {  // NOLINT
  using Matrix = Matrix<TScalar, K, K>;

  if (k == 1) {
    Matrix matrix = Matrix::Ones(1, 1);
    return matrix;
  } else if (k == 2) {
    Matrix matrix{k, k};
    matrix(0, 0) = TScalar{1};
    matrix(1, 0) = TScalar{0};
    matrix(0, 1) = TScalar{-1};
    matrix(1, 1) = TScalar{1};
    return matrix;
  } else {  // Recursion.
    const auto [Ak, Bk] = coefficients<TScalar, TIndex, K>(k, inputs, idx);
    const auto Mk = nonUniformRecursion<TScalar, TIndex, ((K > 1) ? (K - 1) : Eigen::Dynamic)>(k - 1, inputs, idx);
    Matrix AkMk{k, k};
    Matrix BkMk{k, k};
    AkMk.topLeftCorner(k, k - 1) = Ak * Mk;
    AkMk.col(k - 1).setZero();
    BkMk.col(0).setZero();
    BkMk.topRightCorner(k, k - 1) = Bk * Mk;
    return AkMk + BkMk;
  }
}

}  // namespace

template <typename TScalar, int TOrder>
BasisInterpolator<TScalar, TOrder>::BasisInterpolator(const Index& order) {
  CHECK_LE(0, order);
  setOrder(order);
}

template <typename TScalar, int TOrder>
auto BasisInterpolator<TScalar, TOrder>::setOrder(const Index& order) -> void {
  if constexpr (TOrder < 0) {
    this->mixing_ = Mixing(order);
    this->polynomials_ = Base::Polynomials(order);
  } else {
    CHECK_EQ(order, TOrder);
    this->mixing_ = Mixing(TOrder);
    this->polynomials_ = Base::Polynomials(TOrder);
  }
}

template <typename TScalar, int TOrder>
auto BasisInterpolator<TScalar, TOrder>::layout(bool uniform) const -> TemporalInterpolatorLayout {
  const auto order = this->order();

  if (uniform) {
    if (order % 2 == 0) {
      const auto margin = order / 2;
      return {.outer_size = order, .inner_size = order, .left_margin = margin, .right_margin = margin, .left_padding = 0, .right_padding = 0};
    } else {
      const auto margin = (order - 1) / 2;
      return {.outer_size = order, .inner_size = order, .left_margin = margin + 1, .right_margin = margin, .left_padding = 0, .right_padding = 0};
    }
  } else {
    const auto degree = order - 1;
    const auto padding = (degree - 1) / 2;
    return {.outer_size = 2 * degree, .inner_size = order, .left_margin = degree, .right_margin = degree, .left_padding = padding, .right_padding = padding};
  }
}

template <typename TScalar, int TOrder>
auto BasisInterpolator<TScalar, TOrder>::Mixing(const Index& order) -> OrderMatrix {
  return OrderMatrix::Ones(order, order).template triangularView<Eigen::Upper>() * uniformRecursion<TScalar, Index, TOrder>(order);
}

template <typename TScalar, int TOrder>
auto BasisInterpolator<TScalar, TOrder>::mixing(const TScalar* const* inputs, const Index& idx) const -> OrderMatrix {
  const auto order = this->order();
  return OrderMatrix::Ones(order, order).template triangularView<Eigen::Upper>() * nonUniformRecursion<TScalar, Index, TOrder>(order, inputs, idx);
}

template class BasisInterpolator<double, 4>;
template class BasisInterpolator<double, Eigen::Dynamic>;

}  // namespace hyper::state
