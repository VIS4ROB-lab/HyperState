/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include "hyper/state/interpolators/spatial/cartesian.hpp"
#include "hyper/variables/cartesian.hpp"

namespace hyper::state {

template <typename TScalar, int TSize>
auto CartesianInterpolator<TScalar, TSize>::evaluate(Result<Output>& result, const Eigen::Ref<const MatrixX<Scalar>>& weights, const Scalar* const* inputs, int s_idx, int e_idx,
                                                     int offs) -> void {
  // Input lambda definition.
  auto I = [&inputs, &offs](int i) {
    return Eigen::Map<const Input>{inputs[i] + offs};
  };

  // Compute increments.
  auto values = MatrixNX<Output>{Output::kNumParameters, weights.rows()};
  values.col(0).noalias() = I(s_idx);
  for (auto i = s_idx; i < e_idx; ++i) {
    values.col(i - s_idx + 1).noalias() = I(i + 1) - I(i);
  }

  // Compute value and derivatives.
  if (result.degree() == Derivative::VALUE) {
    result.value() = values * weights.col(Derivative::VALUE);
  } else {
    result.value() = values * weights.col(Derivative::VALUE);
    result.tangents() = values * weights.rightCols(result.degree());
  }

  // Compute Jacobians.
  if (result.hasJacobians()) {
    // Jacobian lambda definition.
    auto J = [&result, &offs](int k, int i) {
      return result.template jacobian<Output::kNumParameters, Input::kNumParameters>(k, i, 0, offs);
    };

    for (auto k = 0; k <= result.degree(); ++k) {
      for (auto i = s_idx; i < e_idx; ++i) {
        J(k, i).diagonal().array() = weights(i - s_idx, k) - weights(i - s_idx + 1, k);
      }
      J(k, e_idx).diagonal().array() = weights(e_idx - s_idx, k);
    }
  }
}

template class SpatialInterpolator<variables::Cartesian<double, 3>>;

}  // namespace hyper::state
