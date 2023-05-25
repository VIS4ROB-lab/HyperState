/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include "hyper/state/interpolators/spatial/su2.hpp"
#include "hyper/variables/su2.hpp"

namespace hyper::state {

using namespace variables;

#if HYPER_COMPILE_WITH_GLOBAL_LIE_GROUP_DERIVATIVES
template <typename TScalar>
auto SU2Interpolator<TScalar>::evaluate(Result<Output>& result, const TScalar* weights, const TScalar* const* inputs, int s_idx, int e_idx, int offs) -> void {
  // Definitions.
  using Tangent = variables::Tangent<Output>;
  using Jacobian = hyper::JacobianNM<Tangent>;

  // Map weights.
  const auto n_rows = e_idx - s_idx + 1;
  const auto n_cols = result.degree() + 1;
  const auto W = Eigen::Map<const MatrixX<TScalar>>{weights, n_rows, n_cols};

  // Input lambda definition.
  auto I = [&inputs, &offs](int i) {
    return Eigen::Map<const Input>{inputs[i] + offs};
  };

  // Initialize result.
  result.value() = I(s_idx);
  if (0 < result.degree()) {
    result.velocity().setZero();
    if (1 < result.degree()) {
      result.acceleration().setZero();
    }
  }

  if (!result.hasJacobians()) {
    for (auto i = s_idx; i < e_idx; ++i) {
      const auto j = i + 1;
      const auto k = j - s_idx;

      const auto w_0 = W(k, 0);
      const auto R_ij = I(i).gInv().gPlus(I(j));
      const Tangent d_i = R_ij.gLog();
      const Tangent w_i = w_0 * d_i;
      const auto R_i = w_i.gExp();

      if (0 < result.degree()) {
        const auto w_1 = W(k, 1);
        const Tangent R_d_i = result.value().act(d_i);
        const Tangent v_i = w_1 * R_d_i;
        result.velocity().noalias() += v_i;

        if (1 < result.degree()) {
          const auto w_2 = W(k, 2);
          result.acceleration().noalias() += w_2 * R_d_i + result.velocity().cross(v_i);
        }
      }

      // Update value.
      result.value() *= R_i;
    }
  } else {
    // Jacobian lambda definitions.
    auto Jr = [&result, &offs](int k, int i) {
      return result.template jacobian<Tangent::kNumParameters, Tangent::kNumParameters>(k, i, Tangent::kAngularOffset, Tangent::kAngularOffset + offs);
    };

    // Initialize value Jacobian.
    Jr(0, s_idx).setIdentity();

    for (auto i = s_idx; i < e_idx; ++i) {
      const auto j = i + 1;
      const auto k = j - s_idx;

      Jacobian J_inv, J_i_R_i, J_R_j, J_d_i, J_w_i;
      const auto w_0 = W(k, 0);
      const auto R_ij = I(i).gInv(J_inv.data()).gPlus(I(j), J_i_R_i.data(), J_R_j.data());
      const Tangent d_i = R_ij.gLog(J_d_i.data());
      const Tangent w_i = w_0 * d_i;
      const auto R_i = w_i.gExp(J_w_i.data());

      // Partial value Jacobians.
      const Jacobian J_x_i = result.value() * J_w_i * w_0 * J_d_i;

      // Update left value Jacobian.
      Jr(0, i).noalias() += J_x_i * J_i_R_i * J_inv;

      // Velocity update.
      if (0 < result.degree()) {
        const auto w_1 = W(k, 1);
        const Tangent R_d_i = result.value().act(d_i);
        const Tangent v_i = w_1 * R_d_i;
        result.velocity().noalias() += v_i;

        // Update left velocity Jacobians.
        // ...

        // Propagate velocity updates.
        // ...

        // Acceleration update.
        if (1 < result.degree()) {
          const auto w_2 = W(k, 2);
          result.acceleration().noalias() += w_2 * R_d_i + result.velocity().cross(v_i);

          // Update left acceleration Jacobians.
          // ...

          // Propagate acceleration updates.
          // ...

          // Update right acceleration Jacobians.
          // ...
        }

        // Update right velocity Jacobian.
        // ...
      }

      // Update right value Jacobian.
      Jr(0, j).noalias() = J_x_i * J_R_j;

      // Update value.
      result.value() *= R_i;
    }
  }
}
#else
template <typename TScalar>
auto SU2Interpolator<TScalar>::evaluate(Result<Output>& result, const TScalar* weights, const TScalar* const* inputs, int s_idx, int e_idx, int offs) -> void {
  // Definitions.
  using Tangent = variables::Tangent<Output>;
  using Jacobian = hyper::JacobianNM<Tangent>;

  // Map weights.
  const auto n_rows = e_idx - s_idx + 1;
  const auto n_cols = result.degree() + 1;
  const auto W = Eigen::Map<const MatrixX<TScalar>>{weights, n_rows, n_cols};

  // Input lambda definition.
  auto I = [&inputs, &offs](int i) {
    return Eigen::Map<const Input>{inputs[i] + offs};
  };

  // Initialize result.
  result.value() = Output::Identity();
  if (0 < result.degree()) {
    result.velocity().setZero();
    if (1 < result.degree()) {
      result.acceleration().setZero();
    }
  }

  if (!result.hasJacobians()) {
    for (auto j = e_idx; s_idx < j; --j) {
      const auto i = j - 1;
      const auto k = j - s_idx;

      const auto w_0 = W(k, 0);
      const auto R_ij = I(i).gInv().gPlus(I(j));
      const Tangent d_i = R_ij.gLog();
      const Tangent w_i = w_0 * d_i;
      const auto R_i = w_i.gExp();

      if (0 < result.degree()) {
        const auto w_1 = W(k, 1);
        const auto i_R = result.value().gInv();
        const Tangent i_R_d_i = i_R.act(d_i);
        const Tangent v_i = w_1 * i_R_d_i;
        result.velocity().noalias() += v_i;

        if (1 < result.degree()) {
          const auto w_2 = W(k, 2);
          result.acceleration().noalias() += w_2 * i_R_d_i + v_i.cross(result.velocity());
        }
      }

      // Update value.
      result.value() = R_i * result.value();
    }

    // Update value.
    result.value() = I(s_idx) * result.value();

  } else {
    // Jacobian lambda definitions.
    auto Jr = [&result, &offs](int k, int i) {
      return result.template jacobian<Tangent::kNumParameters, Tangent::kNumParameters>(k, i, Tangent::kAngularOffset, Tangent::kAngularOffset + offs);
    };

    for (auto j = e_idx; s_idx < j; --j) {
      const auto i = j - 1;
      const auto k = j - s_idx;

      Jacobian J_inv, J_i_R_i, J_R_j, J_d_i, J_w_i;
      const auto w_0 = W(k, 0);
      const auto R_ij = I(i).gInv(J_inv.data()).gPlus(I(j), J_i_R_i.data(), J_R_j.data());
      const Tangent d_i = R_ij.gLog(J_d_i.data());
      const Tangent w_i = w_0 * d_i;
      const auto R_i = w_i.gExp(J_w_i.data());

      const auto i_R = result.value().gInv();

      const Jacobian J_i_R = i_R.gAdj();
      const Jacobian J_R_i = J_i_R_i * J_inv;
      const Jacobian J_x_i = (J_i_R * J_w_i * w_0 * J_d_i).eval();

      // Update left value Jacobian.
      Jr(0, i).noalias() = J_x_i * J_R_i;

      // Velocity update.
      if (0 < result.degree()) {
        const auto w_1 = W(k, 1);
        const Tangent i_R_d_i = i_R.act(d_i);
        const Tangent v_i = w_1 * i_R_d_i;
        result.velocity().noalias() += v_i;

        const Jacobian J_i_R_d_i = J_i_R * J_d_i;
        const Jacobian i_R_d_i_x = i_R_d_i.hat();
        const Jacobian J_v_0 = w_1 * J_i_R_d_i;
        const Jacobian J_v_1 = w_1 * i_R_d_i_x;

        // Update left velocity Jacobians.
        Jr(1, i).noalias() = J_v_0 * J_R_i;

        // Propagate velocity updates.
        for (auto n = e_idx; j < n; --n) {
          Jr(1, n).noalias() += J_v_1 * Jr(0, n);
        }

        // Acceleration update.
        if (1 < result.degree()) {
          const auto w_2 = W(k, 2);
          result.acceleration() += w_2 * i_R_d_i + v_i.cross(result.velocity());

          const auto v_i_x = v_i.hat();
          const auto v_x = result.velocity().hat();
          const auto J_a_0 = (w_2 * J_i_R_d_i).eval();
          const auto J_a_1 = (v_i_x - v_x).eval();
          const auto J_a_2 = (w_2 * i_R_d_i_x).eval();
          const auto J_a_3 = (J_a_2 - v_x * J_v_1).eval();

          // Update left acceleration Jacobians.
          Jr(2, i).noalias() = J_a_0 * J_R_i + J_a_1 * Jr(1, i);

          // Propagate acceleration updates.
          for (auto n = e_idx; j < n; --n) {
            Jr(2, n).noalias() += J_a_3 * Jr(0, n) + v_i_x * Jr(1, n);
          }

          // Update right acceleration Jacobians.
          Jr(2, j).noalias() += (J_a_0 + J_a_1 * J_v_0) + (J_a_2 + J_a_1 * J_v_1) * Jr(0, j) + v_i_x * Jr(1, j);
        }

        // Update right velocity Jacobian.
        Jr(1, j).noalias() += J_v_0 + J_v_1 * Jr(0, j);
      }

      // Update right value Jacobian.
      Jr(0, j).noalias() += J_x_i;

      // Update value.
      result.value() = R_i * result.value();
    }

    // Update value Jacobian.
    Jr(0, s_idx).noalias() += result.value().gInv().gAdj();

    // Update value.
    result.value() = I(s_idx) * result.value();
  }
}
#endif

template class SpatialInterpolator<SU2<double>>;

}  // namespace hyper::state
