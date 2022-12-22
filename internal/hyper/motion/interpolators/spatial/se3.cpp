/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include "hyper/motion/interpolators/spatial/se3.hpp"
#include "hyper/variables/adapters.hpp"
#include "hyper/variables/stamped.hpp"

namespace hyper {

auto SpatialInterpolator<Stamped<SE3<Scalar>>>::evaluate(const Weights& weights, const Variables& variables, const Outputs& outputs, const Jacobians& jacobians, const Index& offset, const bool old_jacobians) -> bool {
  const auto derivative = static_cast<MotionDerivative>(weights.cols() - 1);
  switch (derivative) {
    case MotionDerivative::VALUE:
      return evaluate<MotionDerivative::VALUE>(weights, variables, outputs, jacobians, offset, old_jacobians);
    case MotionDerivative::VELOCITY:
      return evaluate<MotionDerivative::VELOCITY>(weights, variables, outputs, jacobians, offset, old_jacobians);
    case MotionDerivative::ACCELERATION:
      return evaluate<MotionDerivative::ACCELERATION>(weights, variables, outputs, jacobians, offset, old_jacobians);
    default:
      LOG(FATAL) << "Requested derivative is not available.";
      return {};
  }
}

template <MotionDerivative TMotionDerivative>
auto SpatialInterpolator<Stamped<SE3<Scalar>>>::evaluate(const Weights& weights, const Variables& variables, const Outputs& outputs, const Jacobians& jacobians, const Index& offset, const bool old_jacobians) -> bool {
  // Definitions.
  using Rotation = typename SE3<Scalar>::Rotation;
  using Translation = typename SE3<Scalar>::Translation;
  using SU2Tangent = hyper::Tangent<SU2<Scalar>>;

  // Constants.
  constexpr auto kValue = MotionDerivative::VALUE;
  constexpr auto kVelocity = MotionDerivative::VELOCITY;
  constexpr auto kAcceleration = MotionDerivative::ACCELERATION;

  const auto num_variables = weights.rows();
  // const auto num_derivatives = weights.cols();

  // Compute indices.
  const auto end_idx = offset + num_variables;
  const auto last_idx = end_idx - 1;

  // Allocate accumulators.
  Rotation R = Rotation::Identity();
  Translation x = Translation::Zero();
  Tangent v = Tangent::Zero();
  Tangent a = Tangent::Zero();

  if (!old_jacobians) {
    // Retrieves first input.
    const auto T_0 = Eigen::Map<const Manifold>{variables[offset]};

    /* Allocate accumulators.
    Rotation R = T_0.rotation();
    Translation x = T_0.translation();
    Tangent v = Tangent::Zero();
    Tangent a = Tangent::Zero();

    for (Index i = offset; i < last_idx; ++i) {
      const auto T_a = Eigen::Map<const Manifold>{inputs[i]}.variable();
      const auto T_b = Eigen::Map<const Manifold>{inputs[i + 1]}.variable();
      const auto w0_i = weights(i - offset + 1, Derivative::VALUE);

      const auto R_ab = T_a.rotation().groupInverse().groupPlus(T_b.rotation());
      const auto d_ab = R_ab.toTangent();
      const auto x_ab = Translation{T_b.translation() - T_a.translation()};
      const auto w_ab = SU2Tangent{w0_i * d_ab};
      const auto R_i = w_ab.toManifold();
      const auto x_i = Translation{w0_i * x_ab};

      R *= R_i;
      x += x_i;

      if constexpr (Derivative::VALUE < TMotionDerivative) {
        const auto i_R_i = R_i.groupInverse();
        const auto w1_i = weights(i - offset + 1, Derivative::VELOCITY);
        v.angular() = i_R_i * v.angular() + w1_i * d_ab;
        v.linear() += w1_i * x_ab;

        if constexpr (Derivative::VELOCITY < TMotionDerivative) {
          const auto w2_i = weights(i - offset + 1, Derivative::ACCELERATION);
          a.angular() = i_R_i * a.angular() + w1_i * v.angular().cross(d_ab) + w2_i * d_ab;
          a.linear() += w2_i * x_ab;
        }
      }
    } */

    for (Index i = last_idx; offset < i; --i) {
      const auto T_a = Eigen::Map<const Manifold>{variables[i - 1]};
      const auto T_b = Eigen::Map<const Manifold>{variables[i]};
      const auto w0_i = weights(i - offset, kValue);

      const auto R_ab = T_a.rotation().groupInverse().groupPlus(T_b.rotation());
      const auto d_ab = R_ab.toTangent();
      const auto x_ab = Translation{T_b.translation() - T_a.translation()};
      const auto w_ab = SU2Tangent{w0_i * d_ab};
      const auto R_i = w_ab.toManifold();
      const auto x_i = Translation{w0_i * x_ab};

      if constexpr (kValue < TMotionDerivative) {
        const auto i_R = R.groupInverse().matrix();
        const auto i_R_d_ab = (i_R * d_ab).eval();
        const auto w1_i = weights(i - offset, kVelocity);
        const auto w1_i_i_R_d_ab = (w1_i * i_R_d_ab).eval();
        v.angular() += w1_i_i_R_d_ab;
        v.linear() += w1_i * x_ab;

        if constexpr (kVelocity < TMotionDerivative) {
          const auto w2_i = weights(i - offset, kAcceleration);
          a.angular() += w2_i * i_R_d_ab - v.angular().cross(w1_i_i_R_d_ab);
          a.linear() += w2_i * x_ab;
        }
      }

      R = R_i * R;
      x = x_i + x;
    }

    R = T_0.rotation() * R;
    x = T_0.translation() + x;

  } else {
    std::vector<JacobianNM<SU2Tangent, SU2<Scalar>>> adapters;
    std::vector<std::vector<Eigen::Map<JacobianNM<SU2Tangent, SU2<Scalar>>, 0, Eigen::OuterStride<Tangent::kNumParameters>>>> Js_r;
    std::vector<std::vector<Eigen::Map<JacobianNM<Tangent::Angular>, 0, Eigen::OuterStride<Tangent::kNumParameters>>>> Js_x;

    adapters.reserve(num_variables);

    for (Index i = 0; i < num_variables; ++i) {
      adapters.emplace_back(SU2JacobianAdapter(variables[i] + Manifold::kRotationOffset));
    }

    Js_r.resize(TMotionDerivative + 1);
    Js_x.resize(TMotionDerivative + 1);
    for (Index k = 0; k < TMotionDerivative + 1; ++k) {
      Js_r.reserve(num_variables);
      Js_x.reserve(num_variables);
      for (Index i = 0; i < num_variables; ++i) {
        adapters.emplace_back(SU2JacobianAdapter(variables[i] + Manifold::kRotationOffset));
        Js_r[k].emplace_back(jacobians[k][i] + Manifold::kRotationOffset * Tangent::kNumParameters + Tangent::kAngularOffset);
        Js_x[k].emplace_back(jacobians[k][i] + Manifold::kTranslationOffset * Tangent::kNumParameters + Tangent::kLinearOffset);
      }
    }

    for (Index i = last_idx; offset < i; --i) {
      const auto T_a = Eigen::Map<const Manifold>{variables[i - 1]};
      const auto T_b = Eigen::Map<const Manifold>{variables[i]};
      const auto w0_i = weights(i - offset, kValue);

      JacobianNM<SU2Tangent> J_R_i_w_ab, J_d_ab_R_ab;
      const auto R_ab = T_a.rotation().groupInverse().groupPlus(T_b.rotation());
      const auto d_ab = R_ab.toTangent(J_d_ab_R_ab.data());
      const auto x_ab = Translation{T_b.translation() - T_a.translation()};
      const auto w_ab = SU2Tangent{w0_i * d_ab};
      const auto R_i = w_ab.toManifold(J_R_i_w_ab.data());
      const auto x_i = Translation{w0_i * x_ab};

      const auto i_R = R.groupInverse().matrix();
      const auto i_R_ab = R_ab.groupInverse().matrix();

      const auto J_x_a = (i_R * J_R_i_w_ab * w0_i * J_d_ab_R_ab).eval();

      // Update left value Jacobian.
      Js_r[kValue][i - 1].noalias() = -J_x_a * i_R_ab * adapters[i - 1];
      Js_x[kValue][i - 1].diagonal().setConstant(-w0_i);

      // Velocity update.
      if constexpr (kValue < TMotionDerivative) {
        const auto i_R_d_ab = (i_R * d_ab).eval();
        const auto i_R_d_ab_x = i_R_d_ab.hat();
        const auto w1_i = weights(i - offset, kVelocity);
        const auto w1_i_i_R_d_ab = (w1_i * i_R_d_ab).eval();

        v.angular() += w1_i_i_R_d_ab;
        v.linear() += w1_i * x_ab;

        const auto J_v_a = (w1_i * i_R * J_d_ab_R_ab).eval();
        const auto J_v_b = (w1_i * i_R_d_ab_x).eval();

        // Update velocity Jacobians.
        Js_r[kVelocity][i - 1].noalias() = -J_v_a * i_R_ab * adapters[i - 1];
        Js_x[kVelocity][i - 1].diagonal().setConstant(-w1_i);
        Js_r[kVelocity][i].noalias() += J_v_a * adapters[i] + J_v_b * Js_r[kValue][i];
        Js_x[kVelocity][i].diagonal().array() += w1_i;

        // Propagate velocity updates.
        for (Index k = last_idx; i < k; --k) {
          Js_r[kVelocity][k].noalias() += J_v_b * Js_r[kValue][k];
        }

        // Acceleration update.
        if constexpr (kVelocity < TMotionDerivative) {
          const auto w2_i = weights(i - offset, kAcceleration);
          const auto w1_i_i_R_d_ab_x = w1_i_i_R_d_ab.hat();

          a.angular() += w2_i * i_R_d_ab + w1_i_i_R_d_ab.cross(v.angular());
          a.linear() += w2_i * x_ab;

          const auto v_x = v.angular().hat();
          const auto J_a_a = (w2_i * i_R * J_d_ab_R_ab).eval();
          const auto J_a_b = (w2_i * i_R_d_ab_x).eval();
          const auto J_a_c = (J_a_b - v_x * J_v_b).eval();

          // Update acceleration Jacobians.
          Js_r[kAcceleration][i - 1].noalias() = -J_a_a * i_R_ab * adapters[i - 1] + (w1_i_i_R_d_ab_x - v_x) * Js_r[kVelocity][i - 1];
          Js_x[kAcceleration][i - 1].diagonal().setConstant(-w2_i);
          Js_r[kAcceleration][i].noalias() += J_a_a * adapters[i] + J_a_b * Js_r[kValue][i] + (w1_i_i_R_d_ab_x - v_x) * Js_r[kVelocity][i];
          Js_x[kAcceleration][i].diagonal().array() += w2_i;

          // Propagate acceleration updates.
          for (Index k = last_idx; i < k; --k) {
            Js_r[kAcceleration][k].noalias() += J_a_c * Js_r[kValue][k] + w1_i_i_R_d_ab_x * Js_r[kVelocity][k];
          }
        }
      }

      // Update right value Jacobian.
      Js_r[kValue][i].noalias() += J_x_a * adapters[i];
      Js_x[kValue][i].diagonal().array() += w0_i;

      // Value update.
      R = R_i * R;
      x = x_i + x;
    }

    Js_r[kValue][offset].noalias() += R.groupInverse().matrix() * adapters[offset];
    Js_x[kValue][offset].diagonal().array() += Scalar{1};

    const auto T_a = Eigen::Map<const Manifold>{variables[offset]};

    R = T_a.rotation() * R;
    x = T_a.translation() + x;
  }

  Eigen::Map<Manifold>{outputs[kValue]} = SE3<Scalar>{R, x};
  if constexpr (kValue < TMotionDerivative) {
    Eigen::Map<Tangent>{outputs[kVelocity]} = v;
    if constexpr (kVelocity < TMotionDerivative) {
      Eigen::Map<Tangent>{outputs[kAcceleration]} = a;
    }
  }

  return true;
}

} // namespace hyper
