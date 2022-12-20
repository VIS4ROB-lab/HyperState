/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <glog/logging.h>

#include "hyper/motion/interpolators/temporal/polynomial.hpp"

namespace hyper {

template <typename TScalar, int TOrder>
auto PolynomialInterpolator<TScalar, TOrder>::Polynomials(const Index& order) -> OrderMatrix {
  OrderMatrix m = OrderMatrix::Zero(order, order);

  m.row(0).setOnes();
  const auto degree = order - 1;
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
auto PolynomialInterpolator<TScalar, TOrder>::evaluate(const Scalar& time, const MotionDerivative& derivative, const Index& offset, const std::vector<Scalar>& timestamps) const -> Weights {
  const auto dt = time - timestamps[offset];
  const auto i_dt = Scalar{1} / (timestamps[offset + 1] - timestamps[offset]);
  const auto ut = dt * i_dt;

  DCHECK_LE(0, ut);
  DCHECK_LE(ut, 1);

  using Polynomial = Matrix<Scalar, TOrder, Eigen::Dynamic>;

  const auto order = this->order();
  const auto num_derivatives = derivative + 1;
  Polynomial polynomial = Polynomial::Zero(order, num_derivatives);

  auto i_dt_i = TScalar{1};
  for (Index i = 0; i < num_derivatives; ++i) {
    if (i < order) {
      auto ut_j = ut;
      polynomial(i, i) = i_dt_i * polynomials_(i, i);
      for (Index j = i + 1; j < order; ++j) {
        polynomial(j, i) = ut_j * i_dt_i * polynomials_(i, j);
        ut_j *= ut;
      }
    }
    i_dt_i *= i_dt;
  }

  if (isUniform()) {
    return mixing_.lazyProduct(polynomial);
  } else {
    return mixing(timestamps).lazyProduct(polynomial);
  }
}

template class PolynomialInterpolator<double, 4>;
template class PolynomialInterpolator<double, Eigen::Dynamic>;

} // namespace hyper
