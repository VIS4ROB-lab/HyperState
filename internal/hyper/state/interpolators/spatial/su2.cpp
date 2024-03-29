/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include "hyper/state/interpolators/spatial/su2.hpp"
#include "hyper/variables/su2.hpp"

namespace hyper::state {

using namespace variables;

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

  // Allocate accumulators.
  Output R = Output::Identity();
  Tangent v = Tangent::Zero();
  Tangent a = Tangent::Zero();

  if (!result.hasJacobians()) {
    // Retrieves first input.
    const auto I_0 = I(s_idx);

    for (auto i = e_idx; s_idx < i; --i) {
      const auto I_a = I(i - 1);
      const auto I_b = I(i);
      const auto w0_i = W(i - s_idx, Derivative::VALUE);

      const auto R_ab = I_a.gInv().gPlus(I_b);
      const auto d_ab = R_ab.gLog();
      const auto w_ab = Tangent{w0_i * d_ab};
      const auto R_i = w_ab.gExp();

      if (Derivative::VALUE < result.degree()) {
        const auto i_R = R.gInv().gAdj();
        const auto i_R_d_ab = (i_R * d_ab).eval();
        const auto w1_i = W(i - s_idx, Derivative::VELOCITY);
        const auto w1_i_i_R_d_ab = (w1_i * i_R_d_ab).eval();

        v += w1_i_i_R_d_ab;

        if (Derivative::VELOCITY < result.degree()) {
          const auto w2_i = W(i - s_idx, Derivative::ACCELERATION);
          a += w2_i * i_R_d_ab + w1_i_i_R_d_ab.cross(v);
        }
      }

      R = R_i * R;
    }

    R = I_0 * R;

  } else {
    // Jacobian lambda definitions.
    auto Jr = [&result, &offs](int k, int i) {
      return result.template jacobian<Tangent::kNumParameters, Tangent::kNumParameters>(k, i, Tangent::kAngularOffset, Tangent::kAngularOffset + offs);
    };

    for (auto i = e_idx; s_idx < i; --i) {
      const auto I_a = I(i - 1);
      const auto I_b = I(i);
      const auto w0_i = W(i - s_idx, Derivative::VALUE);

      Jacobian J_R_i_w_ab, J_d_ab_R_ab;
      const auto R_ab = I_a.gInv().gPlus(I_b);
      const auto d_ab = R_ab.gLog(J_d_ab_R_ab.data());
      const auto w_ab = Tangent{w0_i * d_ab};
      const auto R_i = w_ab.gExp(J_R_i_w_ab.data());

      const auto i_R = R.gInv().gAdj();
      const auto i_R_ab = R_ab.gInv().gAdj();

      const auto J_x_0 = (i_R * J_R_i_w_ab * w0_i * J_d_ab_R_ab).eval();

      // Update left value Jacobian.
      Jr(Derivative::VALUE, i - 1).noalias() = -J_x_0 * i_R_ab;

      // Velocity update.
      if (Derivative::VALUE < result.degree()) {
        const auto i_R_d_ab = (i_R * d_ab).eval();
        const auto i_R_d_ab_x = i_R_d_ab.hat();
        const auto w1_i = W(i - s_idx, Derivative::VELOCITY);
        const auto w1_i_i_R_d_ab = (w1_i * i_R_d_ab).eval();

        v += w1_i_i_R_d_ab;

        const auto i_R_J_d_ab_R_ab = (i_R * J_d_ab_R_ab).eval();
        const auto J_v_0 = (w1_i * i_R_J_d_ab_R_ab).eval();
        const auto J_v_1 = (w1_i * i_R_d_ab_x).eval();

        // Update left velocity Jacobians.
        Jr(Derivative::VELOCITY, i - 1).noalias() = -J_v_0 * i_R_ab;

        // Propagate velocity updates.
        for (auto k = e_idx; i < k; --k) {
          Jr(Derivative::VELOCITY, k).noalias() += J_v_1 * Jr(Derivative::VALUE, k);
        }

        // Acceleration update.
        if (Derivative::VELOCITY < result.degree()) {
          const auto w2_i = W(i - s_idx, Derivative::ACCELERATION);
          const auto w1_i_i_R_d_ab_x = w1_i_i_R_d_ab.hat();

          a += w2_i * i_R_d_ab + w1_i_i_R_d_ab.cross(v);

          const auto v_x = v.hat();
          const auto J_a_0 = (w2_i * i_R_J_d_ab_R_ab).eval();
          const auto J_a_1 = (w1_i_i_R_d_ab_x - v_x).eval();
          const auto J_a_2 = (w2_i * i_R_d_ab_x).eval();
          const auto J_a_3 = (J_a_2 - v_x * J_v_1).eval();

          // Update acceleration Jacobians.
          Jr(Derivative::ACCELERATION, i - 1).noalias() = -J_a_0 * i_R_ab + J_a_1 * Jr(Derivative::VELOCITY, i - 1);
          Jr(Derivative::ACCELERATION, i).noalias() += (J_a_0 + J_a_1 * J_v_0) + (J_a_2 + J_a_1 * J_v_1) * Jr(Derivative::VALUE, i) + w1_i_i_R_d_ab_x * Jr(Derivative::VELOCITY, i);

          // Propagate acceleration updates.
          for (auto k = e_idx; i < k; --k) {
            Jr(Derivative::ACCELERATION, k).noalias() += J_a_3 * Jr(Derivative::VALUE, k) + w1_i_i_R_d_ab_x * Jr(Derivative::VELOCITY, k);
          }
        }

        // Update right velocity Jacobian.
        Jr(Derivative::VELOCITY, i).noalias() += J_v_0 + J_v_1 * Jr(Derivative::VALUE, i);
      }

      // Update right value Jacobian.
      Jr(Derivative::VALUE, i).noalias() += J_x_0;

      // Value update.
      R = R_i * R;
    }

    Jr(Derivative::VALUE, s_idx).noalias() += R.gInv().gAdj();

    const auto I_a = I(s_idx);
    R = I_a * R;
  }

  result.value() = R;
  if (Derivative::VALUE < result.degree()) {
    result.velocity() = v;
    if (Derivative::VELOCITY < result.degree()) {
      result.acceleration() = a;
    }
  }
}

template class SpatialInterpolator<SU2<double>>;

}  // namespace hyper::state
