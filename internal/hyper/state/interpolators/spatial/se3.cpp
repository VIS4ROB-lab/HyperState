/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include "hyper/state/interpolators/spatial/se3.hpp"
#include "hyper/variables/groups/adapters.hpp"
#include "hyper/variables/stamped.hpp"

namespace hyper::state {

template <typename TScalar>
auto SpatialInterpolator<variables::SE3<TScalar>>::evaluate(const Inputs& inputs, const Weights& weights, bool jacobians, const Index& offset, const Index& stride)
    -> Result<Output> {
  // Definitions.
  using Tangent = variables::Tangent<Input>;
  using Rotation = typename Input::Rotation;
  using Translation = typename Input::Translation;
  using Velocity = variables::Tangent<Input>;
  using Acceleration = variables::Tangent<Input>;

  using SU2 = variables::SU2<TScalar>;
  using SU2Tangent = variables::Tangent<SU2>;
  using SU2Jacobian = variables::JacobianNM<SU2Tangent>;
  using SU2Adapter = variables::JacobianNM<SU2Tangent, SU2>;

  // Constants.
  constexpr auto kValue = 0;
  constexpr auto kVelocity = 1;
  constexpr auto kAcceleration = 2;

  const auto num_variables = weights.rows();
  const auto num_derivatives = weights.cols();

  const auto degree = num_derivatives - 1;
  const auto idx = offset + num_variables - 1;

  // Allocate result.
  auto result = Result<Output>(degree, jacobians, inputs.size(), stride);

  // Allocate accumulators.
  Rotation R = Rotation::Identity();
  Translation x = Translation::Zero();
  Velocity v = Velocity::Zero();
  Acceleration a = Acceleration::Zero();

  if (!jacobians) {
    // Retrieves first input.
    const auto T_0 = Eigen::Map<const Input>{inputs[offset]};

    for (Index i = idx; offset < i; --i) {
      const auto T_a = Eigen::Map<const Input>{inputs[i - 1]};
      const auto T_b = Eigen::Map<const Input>{inputs[i]};
      const auto w0_i = weights(i - offset, kValue);

      const auto R_ab = T_a.rotation().groupInverse().groupPlus(T_b.rotation());
      const auto d_ab = R_ab.toTangent();
      const auto x_ab = Translation{T_b.translation() - T_a.translation()};
      const auto w_ab = SU2Tangent{w0_i * d_ab};
      const auto R_i = w_ab.toManifold();
      const auto x_i = Translation{w0_i * x_ab};

      if (kValue < degree) {
        const auto i_R = R.groupInverse().matrix();
        const auto i_R_d_ab = (i_R * d_ab).eval();
        const auto w1_i = weights(i - offset, kVelocity);
        const auto w1_i_i_R_d_ab = (w1_i * i_R_d_ab).eval();

        v.angular() += w1_i_i_R_d_ab;
        v.linear() += w1_i * x_ab;

        if (kVelocity < degree) {
          const auto w2_i = weights(i - offset, kAcceleration);
          a.angular() += w2_i * i_R_d_ab + w1_i_i_R_d_ab.cross(v.angular());
          a.linear() += w2_i * x_ab;
        }
      }

      R = R_i * R;
      x = x_i + x;
    }

    R = T_0.rotation() * R;
    x = T_0.translation() + x;

  } else {
    // Definitions.
    auto Js_r = [&result](const Index& k, const Index& i, const Index& offset = 0) {
      return result.template jacobian<Tangent::Angular::kNumParameters, Rotation::kNumParameters>(k, i, Tangent::kAngularOffset, Input::kRotationOffset + offset);
    };

    auto Js_x = [&result](const Index& k, const Index& i, const Index& offset = 0) {
      return result.template jacobian<Tangent::Linear::kNumParameters, Translation::kNumParameters>(k, i, Tangent::kLinearOffset, Input::kTranslationOffset + offset);
    };

    std::vector<SU2Adapter> J_adapters;
    J_adapters.reserve(num_variables);
    for (Index i = 0; i < num_variables; ++i) {
      J_adapters.emplace_back(variables::JacobianAdapter<SU2>(inputs[i] + Input::kRotationOffset));
    }

    for (Index i = idx; offset < i; --i) {
      const auto T_a = Eigen::Map<const Input>{inputs[i - 1]};
      const auto T_b = Eigen::Map<const Input>{inputs[i]};
      const auto w0_i = weights(i - offset, kValue);

      SU2Jacobian J_R_i_w_ab, J_d_ab_R_ab;
      const auto R_ab = T_a.rotation().groupInverse().groupPlus(T_b.rotation());
      const auto d_ab = R_ab.toTangent(J_d_ab_R_ab.data());
      const auto x_ab = Translation{T_b.translation() - T_a.translation()};
      const auto w_ab = SU2Tangent{w0_i * d_ab};
      const auto R_i = w_ab.toManifold(J_R_i_w_ab.data());
      const auto x_i = Translation{w0_i * x_ab};

      const auto i_R = R.groupInverse().matrix();
      const auto i_R_ab = R_ab.groupInverse().matrix();

      const auto J_x_0 = (i_R * J_R_i_w_ab * w0_i * J_d_ab_R_ab).eval();

      // Update left value Jacobian.
      Js_r(kValue, i - 1).noalias() = -J_x_0 * i_R_ab * J_adapters[i - 1];
      Js_x(kValue, i - 1).diagonal().setConstant(-w0_i);

      // Velocity update.
      if (kValue < degree) {
        const auto i_R_d_ab = (i_R * d_ab).eval();
        const auto i_R_d_ab_x = i_R_d_ab.hat();
        const auto w1_i = weights(i - offset, kVelocity);
        const auto w1_i_i_R_d_ab = (w1_i * i_R_d_ab).eval();

        v.angular() += w1_i_i_R_d_ab;
        v.linear() += w1_i * x_ab;

        const auto i_R_J_d_ab_R_ab = (i_R * J_d_ab_R_ab).eval();
        const auto J_v_0 = (w1_i * i_R_J_d_ab_R_ab).eval();
        const auto J_v_1 = (w1_i * i_R_d_ab_x).eval();

        // Update left velocity Jacobians.

        Js_r(kVelocity, i - 1).noalias() = -J_v_0 * i_R_ab * J_adapters[i - 1];
        Js_x(kVelocity, i - 1).diagonal().setConstant(-w1_i);

        // Propagate velocity updates.
        for (Index k = idx; i < k; --k) {
          Js_r(kVelocity, k).noalias() += J_v_1 * Js_r(kValue, k);
        }

        // Acceleration update.
        if (kVelocity < degree) {
          const auto w2_i = weights(i - offset, kAcceleration);
          const auto w1_i_i_R_d_ab_x = w1_i_i_R_d_ab.hat();

          a.angular() += w2_i * i_R_d_ab + w1_i_i_R_d_ab.cross(v.angular());
          a.linear() += w2_i * x_ab;

          const auto v_x = v.angular().hat();
          const auto J_a_0 = (w2_i * i_R_J_d_ab_R_ab).eval();
          const auto J_a_1 = (w1_i_i_R_d_ab_x - v_x).eval();
          const auto J_a_2 = (w2_i * i_R_d_ab_x).eval();
          const auto J_a_3 = (J_a_2 - v_x * J_v_1).eval();

          // Update acceleration Jacobians.
          Js_r(kAcceleration, i - 1).noalias() = -J_a_0 * i_R_ab * J_adapters[i - 1] + J_a_1 * Js_r(kVelocity, i - 1);
          Js_x(kAcceleration, i - 1).diagonal().setConstant(-w2_i);
          Js_r(kAcceleration, i).noalias() += (J_a_0 + J_a_1 * J_v_0) * J_adapters[i] + (J_a_2 + J_a_1 * J_v_1) * Js_r(kValue, i) + w1_i_i_R_d_ab_x * Js_r(kVelocity, i);
          Js_x(kAcceleration, i).diagonal().array() += w2_i;

          // Propagate acceleration updates.
          for (Index k = idx; i < k; --k) {
            Js_r(kAcceleration, k).noalias() += J_a_3 * Js_r(kValue, k) + w1_i_i_R_d_ab_x * Js_r(kVelocity, k);
          }
        }

        // Update right velocity Jacobian.
        Js_r(kVelocity, i).noalias() += J_v_0 * J_adapters[i] + J_v_1 * Js_r(kValue, i);
        Js_x(kVelocity, i).diagonal().array() += w1_i;
      }

      // Update right value Jacobian.
      Js_r(kValue, i).noalias() += J_x_0 * J_adapters[i];
      Js_x(kValue, i).diagonal().array() += w0_i;

      // Value update.
      R = R_i * R;
      x = x_i + x;
    }

    Js_r(kValue, offset).noalias() += R.groupInverse().matrix() * J_adapters[offset];
    Js_x(kValue, offset).diagonal().array() += TScalar{1};

    const auto T_a = Eigen::Map<const Input>{inputs[offset]};

    R = T_a.rotation() * R;
    x = T_a.translation() + x;
  }

  result.value() = {R, x};
  if (kValue < degree) {
    result.velocity() = v;
    if (kVelocity < degree) {
      result.acceleration() = a;
    }
  }

  return result;
}

template class SpatialInterpolator<variables::SE3<double>>;

}  // namespace hyper::state
