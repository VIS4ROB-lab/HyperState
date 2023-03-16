/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include "hyper/state/interpolators/spatial/cartesian.hpp"
#include "hyper/variables/cartesian.hpp"

namespace hyper::state {

template <typename TScalar, int TSize>
auto CartesianInterpolator<TScalar, TSize>::evaluate(Result<Output>& result, const TScalar* weights, const TScalar* const* inputs, int s_idx, int e_idx, int offs) -> void {
  // Map weights.
  const auto n_rows = e_idx - s_idx + 1;
  const auto n_cols = result.degree() + 1;
  const auto W = Eigen::Map<const MatrixX<TScalar>>{weights, n_rows, n_cols};

  // Input lambda definition.
  auto I = [&inputs, &offs](int i) {
    return Eigen::Map<const Input>{inputs[i] + offs};
  };

  // Compute increments.
  auto values = MatrixNX<Output>{Output::kNumParameters, W.rows()};
  values.col(0).noalias() = I(s_idx);
  for (auto i = s_idx; i < e_idx; ++i) {
    values.col(i - s_idx + 1).noalias() = I(i + 1) - I(i);
  }

  // Compute value and derivatives.
  if (result.degree() == Derivative::VALUE) {
    result.value() = values * W.col(Derivative::VALUE);
  } else {
    result.value() = values * W.col(Derivative::VALUE);
    result.tangents() = values * W.rightCols(result.degree());
  }

  // Compute Jacobians.
  if (result.hasJacobians()) {
    // Jacobian lambda definition.
    auto J = [&result, &offs](int k, int i) {
      return result.template jacobian<Output::kNumParameters, Input::kNumParameters>(k, i, 0, offs);
    };

    for (auto k = 0; k <= result.degree(); ++k) {
      for (auto i = s_idx; i < e_idx; ++i) {
        J(k, i).diagonal().array() = W(i - s_idx, k) - W(i - s_idx + 1, k);
      }
      J(k, e_idx).diagonal().array() = W(e_idx - s_idx, k);
    }
  }
}

template class SpatialInterpolator<variables::Cartesian<double, 3>>;

}  // namespace hyper::state
