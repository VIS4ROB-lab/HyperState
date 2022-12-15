/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <glog/logging.h>

#include "hyper/state/interpolators/temporal/polynomial.hpp"

namespace hyper {

namespace {

/// Computes x^k.
template <typename TScalar, typename TIndex>
auto power(const TScalar x, const TIndex& k) -> TScalar {
  auto n = k;
  auto y = TScalar{1};
  while (n) {
    if (n & TIndex{1}) {
      y *= x;
      --n;
    } else {
      y *= x * x;
      n -= 2;
    }
  }
  return y;
}

} // namespace

template <typename TScalar, int TOrder>
auto PolynomialInterpolator<TScalar, TOrder>::isUniform() const -> bool {
  return is_uniform_;
}

template <typename TScalar, int TOrder>
auto PolynomialInterpolator<TScalar, TOrder>::setUniform() -> void {
  is_uniform_ = true;
}

template <typename TScalar, int TOrder>
auto PolynomialInterpolator<TScalar, TOrder>::setNonUniform() -> void {
  is_uniform_ = false;
}

template <typename TScalar, int TOrder>
auto PolynomialInterpolator<TScalar, TOrder>::order() const -> Index {
  return mixing_.rows();
}

template <typename TScalar, int TOrder>
auto PolynomialInterpolator<TScalar, TOrder>::evaluate(const Query& query) const -> bool {
  // Unpack query.
  const auto& [time, derivative, times, weights] = query;

  // Normalize time.
  const auto index = (times.size() - 1) / 2;
  const auto t0 = times[index];
  const auto t1 = times[index + 1];

  const auto i_ut = Scalar{1} / (t1 - t0);
  const auto ut = (time - t0) * i_ut;

  // Sanity checks.
  DCHECK_LE(0, ut);
  DCHECK_LE(ut, 1);

  // Allocate weights.
  const auto order = this->order();
  auto W = Eigen::Map<Weights>{weights, order, derivative + 1};

  if (isUniform()) {
    for (Index k = 0; k < derivative + 1; ++k) {
      W.col(k).noalias() = mixing_ * polynomial(ut, k) * power(i_ut, k);
    }
  } else {
    const auto mixing = this->mixing(times);
    for (Index k = 0; k < derivative + 1; ++k) {
      W.col(k).noalias() = mixing * polynomial(ut, k) * power(i_ut, k);
    }
  }

  return true;
}

template <typename TScalar, int TOrder>
auto PolynomialInterpolator<TScalar, TOrder>::polynomials() const -> OrderMatrix {
  const auto order = this->order();
  const auto degree = order - 1;

  OrderMatrix m = OrderMatrix::Zero(order, order);

  m.row(0).setOnes();
  auto next = degree;
  for (Index i = 1; i < order; ++i) {
    for (Index j = degree - next; j < order; ++j) {
      m(i, j) = static_cast<Scalar>(next - degree + j) * m(i - 1, j);
    }
    --next;
  }

  return m;
}

template <typename TScalar, int TOrder>
auto PolynomialInterpolator<TScalar, TOrder>::polynomial(const Time& ut, const Index& i) const -> OrderVector {
  const auto order = this->order();

  OrderVector v = OrderVector::Zero(order);

  if (i < order) {
    v(i, 0) = polynomials_(i, i);
    auto ut_j = ut;
    for (Index j = i + 1; j < order; ++j) {
      v(j, 0) = polynomials_(i, j) * ut_j;
      ut_j *= ut;
    }
  }

  return v;
}

template class PolynomialInterpolator<double, 4>;
template class PolynomialInterpolator<double, Eigen::Dynamic>;

} // namespace hyper
