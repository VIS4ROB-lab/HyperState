/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <glog/logging.h>

#include "hyper/state/interpolators/temporal/basis.hpp"

namespace hyper::state {

namespace {

/// Evaluates low-level coefficients.
template <typename TIndex, int TRows, int TCols = TRows != Eigen::Dynamic ? (TRows - 1) : Eigen::Dynamic>
auto uniformCoefficients(const TIndex& k) -> std::pair<Matrix<TRows, TCols>, Matrix<TRows, TCols>> {
  using Matrix = Matrix<TRows, TCols>;

  Matrix Ak = Matrix::Zero(k, k - 1);
  Matrix Bk = Matrix::Zero(k, k - 1);

  for (TIndex i = 0; i < k - 1; ++i) {
    const auto a_i = Scalar{1} / static_cast<Scalar>(k - 1);
    Ak(i, i) = static_cast<Scalar>(i + 1) * a_i;
    Ak(i + 1, i) = static_cast<Scalar>(k - 2 - i) * a_i;
    Bk(i, i) = -a_i;
    Bk(i + 1, i) = a_i;
  }

  return {Ak, Bk};
}

/// Evaluates the weight precursor (i.e. non-cumulative matrix).
template <typename TIndex, int K>
auto uniformRecursion(const TIndex& k) -> Matrix<K, K> {  // NOLINT
  if (k == 1) {
    Matrix<K, K> matrix = Matrix<K, K>::Ones(1, 1);
    return matrix;
  } else if (k == 2) {
    Matrix<K, K> matrix{k, k};
    matrix(0, 0) = Scalar{1};
    matrix(1, 0) = Scalar{0};
    matrix(0, 1) = Scalar{-1};
    matrix(1, 1) = Scalar{1};
    return matrix;
  } else {
    const auto [Ak, Bk] = uniformCoefficients<TIndex, K>(k);
    const auto Mk = uniformRecursion<TIndex, ((K > 1) ? (K - 1) : Eigen::Dynamic)>(k - 1);
    Matrix<K, K> AkMk{k, k};
    Matrix<K, K> BkMk{k, k};
    AkMk.topLeftCorner(k, k - 1) = Ak * Mk;
    AkMk.col(k - 1).setZero();
    BkMk.col(0).setZero();
    BkMk.topRightCorner(k, k - 1) = Bk * Mk;
    return AkMk + BkMk;
  }
}

/// Evaluates low-level coefficients.
template <typename TIndex>
auto coefficient(const TIndex& i, const TIndex& j, const Scalar* const* inputs, const TIndex& idx) -> std::pair<Scalar, Scalar> {
  const auto v2 = inputs[j - 2][idx];
  const auto v1 = inputs[j - 1][idx];
  const auto vi = inputs[i][idx];
  const auto vij = inputs[i + j - 1][idx];
  const auto i_t = Scalar{1} / (vij - vi);
  return {(v2 - vi) * i_t, (v1 - v2) * i_t};
}

/// Evaluates low-level coefficient matrices.
template <typename TIndex, int TRows, int TCols = TRows != Eigen::Dynamic ? (TRows - 1) : Eigen::Dynamic>
auto coefficients(const TIndex& k, const Scalar* const* inputs, const TIndex& idx) -> std::pair<Matrix<TRows, TCols>, Matrix<TRows, TCols>> {
  Matrix<TRows, TCols> Ak = Matrix<TRows, TCols>::Zero(k, k - 1);
  Matrix<TRows, TCols> Bk = Matrix<TRows, TCols>::Zero(k, k - 1);

  for (TIndex i = 0; i < k - 1; ++i) {
    const auto [a, b] = coefficient<TIndex>(i, k, inputs, idx);
    Ak(i, i) = Scalar{1} - a;
    Ak(i + 1, i) = a;
    Bk(i, i) = -b;
    Bk(i + 1, i) = b;
  }

  return {Ak, Bk};
}

/// Evaluates the weight precursor (i.e. non-cumulative matrix).
template <typename TIndex, int K>
auto nonUniformRecursion(const TIndex& k, const Scalar* const* inputs, const TIndex& idx) -> Matrix<K, K> {  // NOLINT
  if (k == 1) {
    Matrix<K, K> matrix = Matrix<K, K>::Ones(1, 1);
    return matrix;
  } else if (k == 2) {
    Matrix<K, K> matrix{k, k};
    matrix(0, 0) = Scalar{1};
    matrix(1, 0) = Scalar{0};
    matrix(0, 1) = Scalar{-1};
    matrix(1, 1) = Scalar{1};
    return matrix;
  } else {  // Recursion.
    const auto [Ak, Bk] = coefficients<TIndex, K>(k, inputs, idx);
    const auto Mk = nonUniformRecursion<TIndex, ((K > 1) ? (K - 1) : Eigen::Dynamic)>(k - 1, inputs, idx);
    Matrix<K, K> AkMk{k, k};
    Matrix<K, K> BkMk{k, k};
    AkMk.topLeftCorner(k, k - 1) = Ak * Mk;
    AkMk.col(k - 1).setZero();
    BkMk.col(0).setZero();
    BkMk.topRightCorner(k, k - 1) = Bk * Mk;
    return AkMk + BkMk;
  }
}

}  // namespace

template <int TOrder>
BasisInterpolator<TOrder>::BasisInterpolator(int order) {
  CHECK_LE(0, order);
  setOrder(order);
}

template <int TOrder>
auto BasisInterpolator<TOrder>::setOrder(int order) -> void {
  if constexpr (TOrder < 0) {
    this->mixing_ = Mixing(order);
    this->polynomials_ = Base::Polynomials(order);
  } else {
    CHECK_EQ(order, TOrder);
    this->mixing_ = Mixing(TOrder);
    this->polynomials_ = Base::Polynomials(TOrder);
  }
}

template <int TOrder>
auto BasisInterpolator<TOrder>::layout(bool uniform) const -> TemporalInterpolatorLayout {
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

template <int TOrder>
auto BasisInterpolator<TOrder>::Mixing(int order) -> OrderMatrix {
  return OrderMatrix::Ones(order, order).template triangularView<Eigen::Upper>() * uniformRecursion<int, TOrder>(order);
}

template <int TOrder>
auto BasisInterpolator<TOrder>::mixing(const Scalar* const* inputs, int idx) const -> OrderMatrix {
  const auto order = this->order();
  return OrderMatrix::Ones(order, order).template triangularView<Eigen::Upper>() * nonUniformRecursion<int, TOrder>(order, inputs, idx);
}

template class BasisInterpolator<4>;
template class BasisInterpolator<Eigen::Dynamic>;

}  // namespace hyper::state
