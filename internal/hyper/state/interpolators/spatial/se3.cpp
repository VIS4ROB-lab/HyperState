/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include "hyper/state/interpolators/spatial/se3.hpp"
#include "hyper/state/interpolators/spatial/cartesian.hpp"
#include "hyper/variables/groups/adapters.hpp"
#include "hyper/variables/stamped.hpp"

namespace hyper::state {

using namespace variables;

template <typename TScalar>
auto SpatialInterpolator<SE3<TScalar>>::evaluate(Result<Output>& result, const Eigen::Ref<const MatrixX<TScalar>>& weights, const TScalar* const* inputs, const Index& s_idx,
                                                 const Index& e_idx, const Index& offs) -> void {
  // Definitions.
  using Rotation = typename Output::Rotation;
  using Translation = typename Output::Translation;
  using Velocity = variables::Tangent<Output>;
  using Acceleration = variables::Tangent<Output>;

  using RotationTangent = variables::Tangent<Rotation>;
  using RotationJacobian = variables::JacobianNM<RotationTangent>;

  // Constants.
  constexpr auto kValue = 0;
  constexpr auto kVelocity = 1;
  constexpr auto kAcceleration = 2;

  // Input lambda definition.
  auto I = [&inputs, &offs](const Index& i) {
    return Eigen::Map<const Input>{inputs[i] + offs};
  };

  // Allocate accumulators.
  Rotation R = Rotation::Identity();
  Translation x = Translation::Zero();
  Velocity v = Velocity::Zero();
  Acceleration a = Acceleration::Zero();

  if (!result.hasJacobians()) {
    // Retrieves first input.
    const auto I_0 = I(s_idx);

    for (Index i = e_idx; s_idx < i; --i) {
      const auto I_a = I(i - 1);
      const auto I_b = I(i);
      const auto w0_i = weights(i - s_idx, kValue);

      const auto R_ab = I_a.rotation().gInv().gPlus(I_b.rotation());
      const auto d_ab = R_ab.gLog();
      const auto x_ab = Translation{I_b.translation() - I_a.translation()};
      const auto w_ab = RotationTangent{w0_i * d_ab};
      const auto R_i = w_ab.gExp();
      const auto x_i = Translation{w0_i * x_ab};

      if (kValue < result.degree()) {
        const auto i_R = R.gInv().gAdj();
        const auto i_R_d_ab = (i_R * d_ab).eval();
        const auto w1_i = weights(i - s_idx, kVelocity);
        const auto w1_i_i_R_d_ab = (w1_i * i_R_d_ab).eval();

        v.angular() += w1_i_i_R_d_ab;
        v.linear() += w1_i * x_ab;

        if (kVelocity < result.degree()) {
          const auto w2_i = weights(i - s_idx, kAcceleration);
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
    auto Jr = [&result, &offs](const Index& k, const Index& i) {
      using Tangent = Tangent<Input>;
      return result.template jacobian<Tangent::Angular::kNumParameters, RotationTangent::kNumParameters>(k, i, Tangent::kAngularOffset, Input::kRotationOffset + offs);
    };

    auto Jq = [&result, &offs](const Index& k, const Index& i) {
      using Tangent = Tangent<Input>;
      return result.template jacobian<Tangent::Angular::kNumParameters, Rotation::kNumParameters>(k, i, Tangent::kAngularOffset, Input::kRotationOffset + offs);
    };

    auto Jt = [&result, &offs](const Index& k, const Index& i) {
      using Tangent = Tangent<Input>;
      return result.template jacobian<Tangent::Linear::kNumParameters, Translation::kNumParameters>(k, i, Tangent::kLinearOffset, Input::kTranslationOffset + offs);
    };

    for (Index i = e_idx; s_idx < i; --i) {
      const auto I_a = I(i - 1);
      const auto I_b = I(i);
      const auto w0_i = weights(i - s_idx, kValue);

      RotationJacobian J_R_i_w_ab, J_d_ab_R_ab;
      const auto R_ab = I_a.rotation().gInv().gPlus(I_b.rotation());
      const auto d_ab = R_ab.gLog(J_d_ab_R_ab.data());
      const auto x_ab = Translation{I_b.translation() - I_a.translation()};
      const auto w_ab = RotationTangent{w0_i * d_ab};
      const auto R_i = w_ab.gExp(J_R_i_w_ab.data());
      const auto x_i = Translation{w0_i * x_ab};

      const auto i_R = R.gInv().gAdj();
      const auto i_R_ab = R_ab.gInv().gAdj();

      const auto J_x_0 = (i_R * J_R_i_w_ab * w0_i * J_d_ab_R_ab).eval();

      // Update left value Jacobian.
      Jr(kValue, i - 1).noalias() = -J_x_0 * i_R_ab;
      Jt(kValue, i - 1).diagonal().array() = -w0_i;

      // Velocity update.
      if (kValue < result.degree()) {
        const auto i_R_d_ab = (i_R * d_ab).eval();
        const auto i_R_d_ab_x = i_R_d_ab.hat();
        const auto w1_i = weights(i - s_idx, kVelocity);
        const auto w1_i_i_R_d_ab = (w1_i * i_R_d_ab).eval();

        v.angular() += w1_i_i_R_d_ab;
        v.linear() += w1_i * x_ab;

        const auto i_R_J_d_ab_R_ab = (i_R * J_d_ab_R_ab).eval();
        const auto J_v_0 = (w1_i * i_R_J_d_ab_R_ab).eval();
        const auto J_v_1 = (w1_i * i_R_d_ab_x).eval();

        // Update left velocity Jacobians.
        Jr(kVelocity, i - 1).noalias() = -J_v_0 * i_R_ab;
        Jt(kVelocity, i - 1).diagonal().array() = -w1_i;

        // Propagate velocity updates.
        for (Index k = e_idx; i < k; --k) {
          Jr(kVelocity, k).noalias() += J_v_1 * Jr(kValue, k);
        }

        // Acceleration update.
        if (kVelocity < result.degree()) {
          const auto w2_i = weights(i - s_idx, kAcceleration);
          const auto w1_i_i_R_d_ab_x = w1_i_i_R_d_ab.hat();

          a.angular() += w2_i * i_R_d_ab + w1_i_i_R_d_ab.cross(v.angular());
          a.linear() += w2_i * x_ab;

          const auto v_x = v.angular().hat();
          const auto J_a_0 = (w2_i * i_R_J_d_ab_R_ab).eval();
          const auto J_a_1 = (w1_i_i_R_d_ab_x - v_x).eval();
          const auto J_a_2 = (w2_i * i_R_d_ab_x).eval();
          const auto J_a_3 = (J_a_2 - v_x * J_v_1).eval();

          // Update acceleration Jacobians.
          Jr(kAcceleration, i - 1).noalias() = -J_a_0 * i_R_ab + J_a_1 * Jr(kVelocity, i - 1);
          Jt(kAcceleration, i - 1).diagonal().array() = -w2_i;
          Jr(kAcceleration, i).noalias() += (J_a_0 + J_a_1 * J_v_0) + (J_a_2 + J_a_1 * J_v_1) * Jr(kValue, i) + w1_i_i_R_d_ab_x * Jr(kVelocity, i);
          Jt(kAcceleration, i).diagonal().array() += w2_i;

          // Propagate acceleration updates.
          for (Index k = e_idx; i < k; --k) {
            Jr(kAcceleration, k).noalias() += J_a_3 * Jr(kValue, k) + w1_i_i_R_d_ab_x * Jr(kVelocity, k);
          }
        }

        // Update right velocity Jacobian.
        Jr(kVelocity, i).noalias() += J_v_0 + J_v_1 * Jr(kValue, i);
        Jt(kVelocity, i).diagonal().array() += w1_i;
      }

      // Update right value Jacobian.
      Jr(kValue, i).noalias() += J_x_0;
      Jt(kValue, i).diagonal().array() += w0_i;

      // Value update.
      R = R_i * R;
      x = x_i + x;
    }

    Jr(kValue, s_idx).noalias() += R.gInv().gAdj();
    Jt(kValue, s_idx).diagonal().array() += TScalar{1};

    // Apply Jacobian adapters.
    for (Index i = s_idx; i <= e_idx; ++i) {
      const auto Ja = JacobianAdapter<Rotation>(inputs[i] + Input::kRotationOffset);
      for (Index k = 0; k <= result.degree(); ++k) {
        Jq(k, i) = Jr(k, i) * Ja;
      }
    }

    const auto I_a = I(s_idx);
    R = I_a.rotation() * R;
    x = I_a.translation() + x;
  }

  result.value = Output{R, x};
  if (kValue < result.degree()) {
    result.derivative(kVelocity - 1) = v;
    if (kVelocity < result.degree()) {
      result.derivative(kAcceleration - 1) = a;
    }
  }
}

template <typename TScalar>
auto SpatialInterpolator<SE3<TScalar>, Tangent<SE3<TScalar>>>::evaluate(Result<Output>& result, const Eigen::Ref<const MatrixX<TScalar>>& weights, const TScalar* const* inputs,
                                                                        const Index& s_idx, const Index& e_idx, const Index& offs) -> void {
  // Definitions.
  using Rotation = typename Output::Rotation;
  using Translation = typename Output::Translation;
  using Velocity = variables::Tangent<Output>;
  using Acceleration = variables::Tangent<Output>;

  using RotationTangent = variables::Tangent<Rotation>;
  using RotationJacobian = variables::JacobianNM<RotationTangent>;

  // Constants.
  constexpr auto kValue = 0;
  constexpr auto kVelocity = 1;
  constexpr auto kAcceleration = 2;

  // Input lambda definition.
  auto I = [&inputs, &offs](const Index& i) {
    return Eigen::Map<const Input>{inputs[i] + offs};
  };

  // Allocate accumulators.
  Rotation R = Rotation::Identity();
  Translation x = Translation::Zero();
  Velocity v = Velocity::Zero();
  Acceleration a = Acceleration::Zero();

  if (!result.hasJacobians()) {
    // Retrieves first input.
    const auto I_0 = I(s_idx).gExp();

    for (Index i = e_idx; s_idx < i; --i) {
      const auto I_a = I(i - 1);
      const auto I_b = I(i);
      const auto w0_i = weights(i - s_idx, kValue);

      const auto R_ab = I_a.angular().gExp().gInv().gPlus(I_b.angular().gExp());
      const auto d_ab = R_ab.gLog();
      const auto x_ab = Translation{I_b.linear() - I_a.linear()};
      const auto w_ab = RotationTangent{w0_i * d_ab};
      const auto R_i = w_ab.gExp();
      const auto x_i = Translation{w0_i * x_ab};

      if (kValue < result.degree()) {
        const auto i_R = R.gInv().gAdj();
        const auto i_R_d_ab = (i_R * d_ab).eval();
        const auto w1_i = weights(i - s_idx, kVelocity);
        const auto w1_i_i_R_d_ab = (w1_i * i_R_d_ab).eval();

        v.angular() += w1_i_i_R_d_ab;
        v.linear() += w1_i * x_ab;

        if (kVelocity < result.degree()) {
          const auto w2_i = weights(i - s_idx, kAcceleration);
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
    auto Jr = [&result, &offs](const Index& k, const Index& i) {
      return result.template jacobian<Input::Angular::kNumParameters, RotationTangent::kNumParameters>(k, i, Input::kAngularOffset, Input::kAngularOffset + offs);
    };

    auto Jt = [&result, &offs](const Index& k, const Index& i) {
      return result.template jacobian<Input::Linear::kNumParameters, Translation::kNumParameters>(k, i, Input::kLinearOffset, Input::kLinearOffset + offs);
    };

    for (Index i = e_idx; s_idx < i; --i) {
      const auto I_a = I(i - 1);
      const auto I_b = I(i);
      const auto w0_i = weights(i - s_idx, kValue);

      RotationJacobian J_R_i_w_ab, J_d_ab_R_ab;
      const auto R_ab = I_a.angular().gExp().gInv().gPlus(I_b.angular().gExp());
      const auto d_ab = R_ab.gLog(J_d_ab_R_ab.data());
      const auto x_ab = Translation{I_b.linear() - I_a.linear()};
      const auto w_ab = RotationTangent{w0_i * d_ab};
      const auto R_i = w_ab.gExp(J_R_i_w_ab.data());
      const auto x_i = Translation{w0_i * x_ab};

      const auto i_R = R.gInv().gAdj();
      const auto i_R_ab = R_ab.gInv().gAdj();

      const auto J_x_0 = (i_R * J_R_i_w_ab * w0_i * J_d_ab_R_ab).eval();

      // Update left value Jacobian.
      Jr(kValue, i - 1).noalias() = -J_x_0 * i_R_ab;
      Jt(kValue, i - 1).diagonal().array() = -w0_i;

      // Velocity update.
      if (kValue < result.degree()) {
        const auto i_R_d_ab = (i_R * d_ab).eval();
        const auto i_R_d_ab_x = i_R_d_ab.hat();
        const auto w1_i = weights(i - s_idx, kVelocity);
        const auto w1_i_i_R_d_ab = (w1_i * i_R_d_ab).eval();

        v.angular() += w1_i_i_R_d_ab;
        v.linear() += w1_i * x_ab;

        const auto i_R_J_d_ab_R_ab = (i_R * J_d_ab_R_ab).eval();
        const auto J_v_0 = (w1_i * i_R_J_d_ab_R_ab).eval();
        const auto J_v_1 = (w1_i * i_R_d_ab_x).eval();

        // Update left velocity Jacobians.
        Jr(kVelocity, i - 1).noalias() = -J_v_0 * i_R_ab;
        Jt(kVelocity, i - 1).diagonal().array() = -w1_i;

        // Propagate velocity updates.
        for (Index k = e_idx; i < k; --k) {
          Jr(kVelocity, k).noalias() += J_v_1 * Jr(kValue, k);
        }

        // Acceleration update.
        if (kVelocity < result.degree()) {
          const auto w2_i = weights(i - s_idx, kAcceleration);
          const auto w1_i_i_R_d_ab_x = w1_i_i_R_d_ab.hat();

          a.angular() += w2_i * i_R_d_ab + w1_i_i_R_d_ab.cross(v.angular());
          a.linear() += w2_i * x_ab;

          const auto v_x = v.angular().hat();
          const auto J_a_0 = (w2_i * i_R_J_d_ab_R_ab).eval();
          const auto J_a_1 = (w1_i_i_R_d_ab_x - v_x).eval();
          const auto J_a_2 = (w2_i * i_R_d_ab_x).eval();
          const auto J_a_3 = (J_a_2 - v_x * J_v_1).eval();

          // Update acceleration Jacobians.
          Jr(kAcceleration, i - 1).noalias() = -J_a_0 * i_R_ab + J_a_1 * Jr(kVelocity, i - 1);
          Jt(kAcceleration, i - 1).diagonal().array() = -w2_i;
          Jr(kAcceleration, i).noalias() += (J_a_0 + J_a_1 * J_v_0) + (J_a_2 + J_a_1 * J_v_1) * Jr(kValue, i) + w1_i_i_R_d_ab_x * Jr(kVelocity, i);
          Jt(kAcceleration, i).diagonal().array() += w2_i;

          // Propagate acceleration updates.
          for (Index k = e_idx; i < k; --k) {
            Jr(kAcceleration, k).noalias() += J_a_3 * Jr(kValue, k) + w1_i_i_R_d_ab_x * Jr(kVelocity, k);
          }
        }

        // Update right velocity Jacobian.
        Jr(kVelocity, i).noalias() += J_v_0 + J_v_1 * Jr(kValue, i);
        Jt(kVelocity, i).diagonal().array() += w1_i;
      }

      // Update right value Jacobian.
      Jr(kValue, i).noalias() += J_x_0;
      Jt(kValue, i).diagonal().array() += w0_i;

      // Value update.
      R = R_i * R;
      x = x_i + x;
    }

    Jr(kValue, s_idx).noalias() += R.gInv().gAdj();
    Jt(kValue, s_idx).diagonal().array() += TScalar{1};

    // Apply Jacobian adapters.
    for (Index i = s_idx; i <= e_idx; ++i) {
      RotationJacobian J_i = RotationJacobian::Identity();
      I(i).angular().gExp(J_i.data());
      for (Index k = 0; k <= result.degree(); ++k) {
        Jr(k, i) = Jr(k, i) * J_i;
      }
    }

    const auto I_a = I(s_idx).gExp();
    R = I_a.rotation() * R;
    x = I_a.translation() + x;
  }

  result.value = Output{R, x};
  if (kValue < result.degree()) {
    result.derivative(kVelocity - 1) = v;
    if (kVelocity < result.degree()) {
      result.derivative(kAcceleration - 1) = a;
    }
  }
}

template class SpatialInterpolator<SE3<double>>;
template class SpatialInterpolator<SE3<double>, Tangent<SE3<double>>>;

}  // namespace hyper::state
