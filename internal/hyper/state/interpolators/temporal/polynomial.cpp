/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <glog/logging.h>

#include "hyper/state/interpolators/temporal/polynomial.hpp"

namespace hyper::state {

template <int TOrder>
auto PolynomialInterpolator<TOrder>::Polynomials(int order) -> OrderMatrix {
  OrderMatrix m = OrderMatrix::Zero(order, order);

  m.row(0).setOnes();
  const auto degree = order - 1;
  auto next = degree;

  for (auto i = 1; i < order; ++i) {
    for (auto j = degree - next; j < order; ++j) {
      m(i, j) = static_cast<Scalar>(next - degree + j) * m(i - 1, j);
    }
    --next;
  }

  return m;
}

template <int TOrder>
auto PolynomialInterpolator<TOrder>::order() const -> int {
  return mixing_.rows();
}

template <int TOrder>
auto PolynomialInterpolator<TOrder>::evaluate(const Scalar& ut, const Scalar& i_dt, int derivative, const Scalar* const* inputs, int idx) const -> MatrixX {
  DCHECK_LE(0, ut);
  DCHECK_LE(ut, 1);

  using Polynomial = Matrix<TOrder, Eigen::Dynamic>;

  const auto order = this->order();
  const auto num_derivatives = derivative + 1;
  Polynomial polynomial = Polynomial::Zero(order, num_derivatives);

  auto i_dt_i = Scalar{1};
  for (auto i = 0; i < num_derivatives; ++i) {
    if (i < order) {
      auto ut_j = ut;
      polynomial(i, i) = i_dt_i * polynomials_(i, i);
      for (auto j = i + 1; j < order; ++j) {
        polynomial(j, i) = ut_j * i_dt_i * polynomials_(i, j);
        ut_j *= ut;
      }
    }
    i_dt_i *= i_dt;
  }

  if (inputs) {
    return mixing(inputs, idx).lazyProduct(polynomial);
  } else {
    return mixing_.lazyProduct(polynomial);
  }
}

template class PolynomialInterpolator<4>;
template class PolynomialInterpolator<Eigen::Dynamic>;

}  // namespace hyper::state
