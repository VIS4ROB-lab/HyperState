/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <glog/logging.h>

#include "hyper/motion/interpolators/spatial/abstract.hpp"
#include "hyper/motion/interpolators/temporal/basis.hpp"

namespace hyper {

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
auto uniformRecursion(const TIndex& k) -> Matrix<TScalar, K, K> { // NOLINT
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
auto coefficient(const std::vector<TScalar>& v, const TIndex& i, const TIndex& j) -> std::pair<TScalar, TScalar> {
  const auto v2 = v[j - 2];
  const auto v1 = v[j - 1];
  const auto vi = v[i];
  const auto vij = v[i + j - 1];
  const auto i_t = TScalar{1} / (vij - vi);
  return {(v2 - vi) * i_t, (v1 - v2) * i_t};
}

/// Evaluates low-level coefficient matrices.
template <typename TScalar, typename TIndex, int TRows, int TCols = TRows != Eigen::Dynamic ? (TRows - 1) : Eigen::Dynamic>
auto coefficients(const std::vector<TScalar>& v, const TIndex& k) -> std::pair<Matrix<TScalar, TRows, TCols>, Matrix<TScalar, TRows, TCols>> {
  using Matrix = Matrix<TScalar, TRows, TCols>;

  Matrix Ak = Matrix::Zero(k, k - 1);
  Matrix Bk = Matrix::Zero(k, k - 1);

  for (TIndex i = 0; i < k - 1; ++i) {
    const auto [a, b] = coefficient<TScalar, TIndex>(v, i, k);
    Ak(i, i) = TScalar{1} - a;
    Ak(i + 1, i) = a;
    Bk(i, i) = -b;
    Bk(i + 1, i) = b;
  }

  return {Ak, Bk};
}

/// Evaluates the weight precursor (i.e. non-cumulative matrix).
template <typename TScalar, typename TIndex, int K>
auto nonUniformRecursion(const std::vector<TScalar>& v, const TIndex& k) -> Matrix<TScalar, K, K> { // NOLINT
  using Matrix = Matrix<TScalar, K, K>;

  if (k == 1) {
    Matrix matrix = Matrix::Ones(1, 1);
    return matrix;
  } else if (k == 2) {
    Matrix matrix{k, k};
    matrix(0, 0) = Scalar{1};
    matrix(1, 0) = Scalar{0};
    matrix(0, 1) = Scalar{-1};
    matrix(1, 1) = Scalar{1};
    return matrix;
  } else { // Recursion.
    const auto [Ak, Bk] = coefficients<TScalar, TIndex, K>(v, k);
    const auto Mk = nonUniformRecursion<TScalar, TIndex, ((K > 1) ? (K - 1) : Eigen::Dynamic)>(v, k - 1);
    Matrix AkMk{k, k};
    Matrix BkMk{k, k};
    AkMk.topLeftCorner(k, k - 1) = Ak * Mk;
    AkMk.col(k - 1).setZero();
    BkMk.col(0).setZero();
    BkMk.topRightCorner(k, k - 1) = Bk * Mk;
    return AkMk + BkMk;
  }
}

} // namespace

template <typename TScalar, int TOrder>
BasisInterpolator<TScalar, TOrder>::BasisInterpolator(const Index& order) {
  setOrder(order);
}

template <typename TScalar, int TOrder>
auto BasisInterpolator<TScalar, TOrder>::setOrder(const Index& order) -> void {
  if (TOrder < 0) {
    this->mixing_ = Mixing(order);
    this->polynomials_ = Base::Polynomials(order);
  } else {
    CHECK_EQ(order, TOrder);
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
  return OrderMatrix::Ones(order, order).template triangularView<Eigen::Upper>() * uniformRecursion<Scalar, Index, TOrder>(order);
}

template <typename TScalar, int TOrder>
auto BasisInterpolator<TScalar, TOrder>::mixing(const std::vector<Scalar>& times) const -> OrderMatrix {
  const auto order = this->order();
  return OrderMatrix::Ones(order, order).template triangularView<Eigen::Upper>() * nonUniformRecursion<Scalar, Index, TOrder>(times, order);
}

template class BasisInterpolator<double, 4>;
template class BasisInterpolator<double, Eigen::Dynamic>;

} // namespace hyper
