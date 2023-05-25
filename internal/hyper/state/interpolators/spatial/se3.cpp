/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include "hyper/state/interpolators/spatial/se3.hpp"
#include "hyper/variables/se3.hpp"

namespace hyper::state {

using namespace variables;

template <typename TScalar>
auto SE3Interpolator<TScalar>::evaluate(Result<Output>& result, const TScalar* weights, const TScalar* const* inputs, int s_idx, int e_idx, int offs) -> void {
  // Definitions.
  using Rotation = typename Output::Rotation;
  using Translation = typename Output::Translation;
  using Tangent = variables::Tangent<Output>;

  using Angular = variables::Tangent<Rotation>;
  using Linear = variables::Tangent<Translation>;
  using AngularJacobian = hyper::JacobianNM<Angular>;
  // using LinearJacobian = hyper::JacobianNM<Linear>;

  // Map weights.
  const auto n_rows = e_idx - s_idx + 1;
  const auto n_cols = result.degree() + 1;
  const auto W = Eigen::Map<const MatrixX<TScalar>>{weights, n_rows, n_cols};

  // Input lambda definition.
  auto I = [&inputs, &offs](int i) {
    return Eigen::Map<const Input>{inputs[i] + offs};
  };

  // Initialize result.
  auto& x = result.value();
  x = Output{Rotation::Identity(), Translation::Zero()};
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
      const auto I_i = I(i);
      const auto I_j = I(j);
      const Rotation R_ij = I(i).rotation().gInv().gPlus(I(j).rotation());
      const Translation x_ij = I_j.translation() - I_i.translation();
      const Angular d_i = R_ij.gLog();
      const Angular w_i = w_0 * d_i;
      const Rotation R_i = w_i.gExp();

      if (0 < result.degree()) {
        const auto w_1 = W(k, 1);
        const auto i_R = x.rotation().gInv();
        const Angular i_R_d_i = i_R.act(d_i);
        const Angular v_i = w_1 * i_R_d_i;

        auto v = result.velocity();
        v.angular().noalias() += v_i;
        v.linear().noalias() += w_1 * x_ij;

        if (1 < result.degree()) {
          const auto w_2 = W(k, 2);
          auto a = result.acceleration();
          a.angular().noalias() += w_2 * i_R_d_i + v_i.cross(result.velocity().angular());
          a.linear().noalias() += w_2 * x_ij;
        }
      }

      // Update value.
      x.rotation() = R_i * x.rotation();
      x.translation() += w_0 * x_ij;
    }

    // Update value.
    const auto I_s = I(s_idx);
    x.rotation() = I_s.rotation() * x.rotation();
    x.translation() += I_s.translation();

  } else {
    // Jacobian lambda definitions.
    auto Jr = [&result, &offs](int k, int i) {
      return result.template jacobian<Angular::kNumParameters, Angular::kNumParameters>(k, i, Tangent::kAngularOffset, Tangent::kAngularOffset + offs);
    };

    auto Jt = [&result, &offs](int k, int i) {
      return result.template jacobian<Linear::kNumParameters, Linear::kNumParameters>(k, i, Tangent::kLinearOffset, Tangent::kLinearOffset + offs);
    };

    for (auto i = e_idx; s_idx < i; --i) {
      const auto I_a = I(i - 1);
      const auto I_b = I(i);
      const auto w0_i = W(i - s_idx, 0);

      AngularJacobian J_R_i_w_ab, J_d_ab_R_ab;
      const auto R_ab = I_a.rotation().gInv().gPlus(I_b.rotation());
      const auto d_ab = R_ab.gLog(J_d_ab_R_ab.data());
      const auto x_ab = Translation{I_b.translation() - I_a.translation()};
      const auto w_ab = Angular{w0_i * d_ab};
      const auto R_i = w_ab.gExp(J_R_i_w_ab.data());
      const auto x_i = Translation{w0_i * x_ab};

      const auto i_R = x.rotation().gInv().gAdj();
      const auto i_R_ab = R_ab.gInv().gAdj();

      const auto J_x_0 = (i_R * J_R_i_w_ab * w0_i * J_d_ab_R_ab).eval();

      // Update left value Jacobian.
      Jr(0, i - 1).noalias() = -J_x_0 * i_R_ab;
      Jt(0, i - 1).diagonal().array() = -w0_i;

      // Velocity update.
      if (0 < result.degree()) {
        const auto i_R_d_ab = (i_R * d_ab).eval();
        const auto i_R_d_ab_x = i_R_d_ab.hat();
        const auto w1_i = W(i - s_idx, 1);
        const auto w1_i_i_R_d_ab = (w1_i * i_R_d_ab).eval();

        auto v = result.velocity();
        v.angular() += w1_i_i_R_d_ab;
        v.linear() += w1_i * x_ab;

        const auto i_R_J_d_ab_R_ab = (i_R * J_d_ab_R_ab).eval();
        const auto J_v_0 = (w1_i * i_R_J_d_ab_R_ab).eval();
        const auto J_v_1 = (w1_i * i_R_d_ab_x).eval();

        // Update left velocity Jacobians.
        Jr(1, i - 1).noalias() = -J_v_0 * i_R_ab;
        Jt(1, i - 1).diagonal().array() = -w1_i;

        // Propagate velocity updates.
        for (auto n = e_idx; i < n; --n) {
          Jr(1, n).noalias() += J_v_1 * Jr(0, n);
        }

        // Acceleration update.
        if (1 < result.degree()) {
          const auto w2_i = W(i - s_idx, 2);
          const auto w1_i_i_R_d_ab_x = w1_i_i_R_d_ab.hat();

          auto a = result.acceleration();
          a.angular() += w2_i * i_R_d_ab + w1_i_i_R_d_ab.cross(v.angular());
          a.linear() += w2_i * x_ab;

          const auto v_x = v.angular().hat();
          const auto J_a_0 = (w2_i * i_R_J_d_ab_R_ab).eval();
          const auto J_a_1 = (w1_i_i_R_d_ab_x - v_x).eval();
          const auto J_a_2 = (w2_i * i_R_d_ab_x).eval();
          const auto J_a_3 = (J_a_2 - v_x * J_v_1).eval();

          // Update acceleration Jacobians.
          Jr(2, i - 1).noalias() = -J_a_0 * i_R_ab + J_a_1 * Jr(1, i - 1);
          Jt(2, i - 1).diagonal().array() = -w2_i;
          Jr(2, i).noalias() += (J_a_0 + J_a_1 * J_v_0) + (J_a_2 + J_a_1 * J_v_1) * Jr(0, i) + w1_i_i_R_d_ab_x * Jr(1, i);
          Jt(2, i).diagonal().array() += w2_i;

          // Propagate acceleration updates.
          for (auto n = e_idx; i < n; --n) {
            Jr(2, n).noalias() += J_a_3 * Jr(0, n) + w1_i_i_R_d_ab_x * Jr(1, n);
          }
        }

        // Update right velocity Jacobian.
        Jr(1, i).noalias() += J_v_0 + J_v_1 * Jr(0, i);
        Jt(1, i).diagonal().array() += w1_i;
      }

      // Update right value Jacobian.
      Jr(0, i).noalias() += J_x_0;
      Jt(0, i).diagonal().array() += w0_i;

      // Update value.
      x.rotation() = R_i * x.rotation();
      x.translation() += x_i;
    }

    // Update value Jacobian.
    Jr(0, s_idx).noalias() += x.rotation().gInv().gAdj();
    Jt(0, s_idx).diagonal().array() += TScalar{1};

    // Update value.
    const auto I_s = I(s_idx);
    x.rotation() = I_s.rotation() * x.rotation();
    x.translation() += I_s.translation();
  }
}

template class SpatialInterpolator<SE3<double>>;

}  // namespace hyper::state
