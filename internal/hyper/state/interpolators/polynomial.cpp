/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <glog/logging.h>

#include "hyper/state/interpolators/polynomial.hpp"

namespace hyper {

namespace {

/// Computes the x^k.
/// \param x Value to compute x^k for.
/// \param k Power to raise value x by.
/// \return x^k.
auto power(const Scalar x, const Index k) -> Scalar {
  auto n = k;
  auto y = Scalar{1};
  while (n) {
    if (n & Index{1}) {
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

auto PolynomialInterpolator::setUniform(const bool uniform) -> void {
  uniform_ = uniform;
}

auto PolynomialInterpolator::evaluate(const Query& query) const -> bool {
  // Unpack query.
  const auto& [time, derivative, times, weights] = query;

  // Normalize time.
  const auto index = (times.size() - 1) / 2;
  const auto t0 = times[index];
  const auto t1 = times[index + 1];
  const auto i_normalized_stamp = Scalar{1} / (t1 - t0);
  const auto normalized_stamp = (time - t0) * i_normalized_stamp;

  // Sanity checks.
  DCHECK_LE(0, normalized_stamp);
  DCHECK_LE(normalized_stamp, 1);

  // Allocate weights.
  auto W = Eigen::Map<Matrix>{weights, order_, derivative + 1};

  if (uniform_) {
    for (Index k = 0; k < derivative + 1; ++k) {
      W.col(k).noalias() = mixing_ * polynomial(normalized_stamp, k) * power(i_normalized_stamp, k);
    }
  } else {
    const auto M = mixing(times);
    for (Index k = 0; k < derivative + 1; ++k) {
      W.col(k).noalias() = M * polynomial(normalized_stamp, k) * power(i_normalized_stamp, k);
    }
  }

  return true;
}

PolynomialInterpolator::PolynomialInterpolator(const bool uniform)
    : uniform_{uniform},
      degree_{},
      order_{},
      mixing_{},
      polynomials_{} {}

auto PolynomialInterpolator::polynomials() const -> Matrix {
  Matrix matrix = Matrix::Zero(order_, order_);
  matrix.row(0).setOnes();

  auto next = degree_;
  for (Index i = 1; i < order_; ++i) {
    for (Index j = degree_ - next; j < order_; ++j) {
      matrix(i, j) = static_cast<Scalar>(next - degree_ + j) * matrix(i - 1, j);
    }
    --next;
  }

  return matrix;
}

auto PolynomialInterpolator::polynomial(const Time& time, const Index i) const -> Matrix {
  Matrix matrix = Matrix::Zero(order_, 1);

  if (i < order_) {
    matrix(i, 0) = polynomials_(i, i);
    auto stamp_k = time;
    for (Index j = i + 1; j < order_; ++j) {
      matrix(j, 0) = polynomials_(i, j) * stamp_k;
      stamp_k *= time;
    }
  }

  return matrix;
}

} // namespace hyper
