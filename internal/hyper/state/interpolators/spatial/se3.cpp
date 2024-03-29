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

  // Map weights.
  const auto n_rows = e_idx - s_idx + 1;
  const auto n_cols = result.degree() + 1;
  const auto W = Eigen::Map<const MatrixX<TScalar>>{weights, n_rows, n_cols};

  // Input lambda definition.
  auto I = [&inputs, &offs](int i) {
    return Eigen::Map<const Input>{inputs[i] + offs};
  };

  // Allocate accumulators.
  Rotation R = Rotation::Identity();
  Translation x = Translation::Zero();
  Tangent v = Tangent::Zero();
  Tangent a = Tangent::Zero();

  if (!result.hasJacobians()) {
    // Retrieves first input.
    const auto I_0 = I(s_idx);

    for (auto i = e_idx; s_idx < i; --i) {
      const auto I_a = I(i - 1);
      const auto I_b = I(i);
      const auto w0_i = W(i - s_idx, Derivative::VALUE);

      const auto R_ab = I_a.rotation().gInv().gPlus(I_b.rotation());
      const auto d_ab = R_ab.gLog();
      const auto x_ab = Translation{I_b.translation() - I_a.translation()};
      const auto w_ab = Angular{w0_i * d_ab};
      const auto R_i = w_ab.gExp();
      const auto x_i = Translation{w0_i * x_ab};

      if (Derivative::VALUE < result.degree()) {
        const auto i_R = R.gInv().gAdj();
        const auto i_R_d_ab = (i_R * d_ab).eval();
        const auto w1_i = W(i - s_idx, Derivative::VELOCITY);
        const auto w1_i_i_R_d_ab = (w1_i * i_R_d_ab).eval();

        v.angular() += w1_i_i_R_d_ab;
        v.linear() += w1_i * x_ab;

        if (Derivative::VELOCITY < result.degree()) {
          const auto w2_i = W(i - s_idx, Derivative::ACCELERATION);
          a.angular() += w2_i * i_R_d_ab + w1_i_i_R_d_ab.cross(v.angular());
          a.linear() += w2_i * x_ab;
        }
      }

      R = R_i * R;
      x = x_i + x;
    }

    R = I_0.rotation() * R;
    x = I_0.translation() + x;

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
      const auto w0_i = W(i - s_idx, Derivative::VALUE);

      AngularJacobian J_R_i_w_ab, J_d_ab_R_ab;
      const auto R_ab = I_a.rotation().gInv().gPlus(I_b.rotation());
      const auto d_ab = R_ab.gLog(J_d_ab_R_ab.data());
      const auto x_ab = Translation{I_b.translation() - I_a.translation()};
      const auto w_ab = Angular{w0_i * d_ab};
      const auto R_i = w_ab.gExp(J_R_i_w_ab.data());
      const auto x_i = Translation{w0_i * x_ab};

      const auto i_R = R.gInv().gAdj();
      const auto i_R_ab = R_ab.gInv().gAdj();

      const auto J_x_0 = (i_R * J_R_i_w_ab * w0_i * J_d_ab_R_ab).eval();

      // Update left value Jacobian.
      Jr(Derivative::VALUE, i - 1).noalias() = -J_x_0 * i_R_ab;
      Jt(Derivative::VALUE, i - 1).diagonal().array() = -w0_i;

      // Velocity update.
      if (Derivative::VALUE < result.degree()) {
        const auto i_R_d_ab = (i_R * d_ab).eval();
        const auto i_R_d_ab_x = i_R_d_ab.hat();
        const auto w1_i = W(i - s_idx, Derivative::VELOCITY);
        const auto w1_i_i_R_d_ab = (w1_i * i_R_d_ab).eval();

        v.angular() += w1_i_i_R_d_ab;
        v.linear() += w1_i * x_ab;

        const auto i_R_J_d_ab_R_ab = (i_R * J_d_ab_R_ab).eval();
        const auto J_v_0 = (w1_i * i_R_J_d_ab_R_ab).eval();
        const auto J_v_1 = (w1_i * i_R_d_ab_x).eval();

        // Update left velocity Jacobians.
        Jr(Derivative::VELOCITY, i - 1).noalias() = -J_v_0 * i_R_ab;
        Jt(Derivative::VELOCITY, i - 1).diagonal().array() = -w1_i;

        // Propagate velocity updates.
        for (auto k = e_idx; i < k; --k) {
          Jr(Derivative::VELOCITY, k).noalias() += J_v_1 * Jr(Derivative::VALUE, k);
        }

        // Acceleration update.
        if (Derivative::VELOCITY < result.degree()) {
          const auto w2_i = W(i - s_idx, Derivative::ACCELERATION);
          const auto w1_i_i_R_d_ab_x = w1_i_i_R_d_ab.hat();

          a.angular() += w2_i * i_R_d_ab + w1_i_i_R_d_ab.cross(v.angular());
          a.linear() += w2_i * x_ab;

          const auto v_x = v.angular().hat();
          const auto J_a_0 = (w2_i * i_R_J_d_ab_R_ab).eval();
          const auto J_a_1 = (w1_i_i_R_d_ab_x - v_x).eval();
          const auto J_a_2 = (w2_i * i_R_d_ab_x).eval();
          const auto J_a_3 = (J_a_2 - v_x * J_v_1).eval();

          // Update acceleration Jacobians.
          Jr(Derivative::ACCELERATION, i - 1).noalias() = -J_a_0 * i_R_ab + J_a_1 * Jr(Derivative::VELOCITY, i - 1);
          Jt(Derivative::ACCELERATION, i - 1).diagonal().array() = -w2_i;
          Jr(Derivative::ACCELERATION, i).noalias() += (J_a_0 + J_a_1 * J_v_0) + (J_a_2 + J_a_1 * J_v_1) * Jr(Derivative::VALUE, i) + w1_i_i_R_d_ab_x * Jr(Derivative::VELOCITY, i);
          Jt(Derivative::ACCELERATION, i).diagonal().array() += w2_i;

          // Propagate acceleration updates.
          for (auto k = e_idx; i < k; --k) {
            Jr(Derivative::ACCELERATION, k).noalias() += J_a_3 * Jr(Derivative::VALUE, k) + w1_i_i_R_d_ab_x * Jr(Derivative::VELOCITY, k);
          }
        }

        // Update right velocity Jacobian.
        Jr(Derivative::VELOCITY, i).noalias() += J_v_0 + J_v_1 * Jr(Derivative::VALUE, i);
        Jt(Derivative::VELOCITY, i).diagonal().array() += w1_i;
      }

      // Update right value Jacobian.
      Jr(Derivative::VALUE, i).noalias() += J_x_0;
      Jt(Derivative::VALUE, i).diagonal().array() += w0_i;

      // Value update.
      R = R_i * R;
      x = x_i + x;
    }

    Jr(Derivative::VALUE, s_idx).noalias() += R.gInv().gAdj();
    Jt(Derivative::VALUE, s_idx).diagonal().array() += TScalar{1};

    const auto I_a = I(s_idx);
    R = I_a.rotation() * R;
    x = I_a.translation() + x;
  }

  result.value() = Output{R, x};
  if (Derivative::VALUE < result.degree()) {
    result.velocity() = v;
    if (Derivative::VELOCITY < result.degree()) {
      result.acceleration() = a;
    }
  }
}

template class SpatialInterpolator<SE3<double>>;

}  // namespace hyper::state
