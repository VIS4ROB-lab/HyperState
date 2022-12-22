/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include "hyper/motion/interpolators/spatial/se3.hpp"
#include "hyper/variables/adapters.hpp"
#include "hyper/variables/stamped.hpp"

namespace hyper {

namespace {

using Input = Stamped<SE3<Scalar>>;
constexpr auto kNumInputParameters = Input::kNumParameters;

template <Index TCols = 3, typename TMatrix>
inline auto RotationJacobian(TMatrix& matrix, const Index& index) {
  return matrix.template block<3, TCols>(0, index * kNumInputParameters + 0);
}

template <typename TMatrix>
inline auto TranslationJacobian(TMatrix& matrix, const Index& index) {
  return matrix.template block<3, 3>(3, index * kNumInputParameters + 4);
}

} // namespace

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

  const auto num_variables = weights.rows();
  // const auto num_derivatives = weights.cols();

  // Allocate result.
  TemporalMotionResult<Scalar> result;
  auto& [xs, Js] = result;
  Js.reserve(TMotionDerivative + 1);

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
    const auto T_0 = Eigen::Map<const Input>{variables[offset]}.variable();

    /* Allocate accumulators.
    Rotation R = T_0.rotation();
    Translation x = T_0.translation();
    Tangent v = Tangent::Zero();
    Tangent a = Tangent::Zero();

    for (Index i = offset; i < last_idx; ++i) {
      const auto T_a = Eigen::Map<const Input>{inputs[i]}.variable();
      const auto T_b = Eigen::Map<const Input>{inputs[i + 1]}.variable();
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
      const auto T_a = Eigen::Map<const Input>{variables[i - 1]}.variable();
      const auto T_b = Eigen::Map<const Input>{variables[i]}.variable();
      const auto w0_i = weights(i - offset, MotionDerivative::VALUE);

      const auto R_ab = T_a.rotation().groupInverse().groupPlus(T_b.rotation());
      const auto d_ab = R_ab.toTangent();
      const auto x_ab = Translation{T_b.translation() - T_a.translation()};
      const auto w_ab = SU2Tangent{w0_i * d_ab};
      const auto R_i = w_ab.toManifold();
      const auto x_i = Translation{w0_i * x_ab};

      if constexpr (MotionDerivative::VALUE < TMotionDerivative) {
        const auto i_R = R.groupInverse().matrix();
        const auto i_R_d_ab = (i_R * d_ab).eval();
        const auto w1_i = weights(i - offset, MotionDerivative::VELOCITY);
        const auto w1_i_i_R_d_ab = (w1_i * i_R_d_ab).eval();
        v.angular() += w1_i_i_R_d_ab;
        v.linear() += w1_i * x_ab;

        if constexpr (MotionDerivative::VELOCITY < TMotionDerivative) {
          const auto w2_i = weights(i - offset, MotionDerivative::ACCELERATION);
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
    // Allocate Jacobians.
    const auto num_parameters = variables.size() * kNumInputParameters;
    Js.emplace_back(JacobianX<Scalar>::Zero(kDimTangent, num_parameters));
    if constexpr (MotionDerivative::VALUE < TMotionDerivative) {
      Js.emplace_back(JacobianX<Scalar>::Zero(kDimTangent, num_parameters));
      if constexpr (MotionDerivative::VELOCITY < TMotionDerivative) {
        Js.emplace_back(JacobianX<Scalar>::Zero(kDimTangent, num_parameters));
      }
    }

    for (Index i = last_idx; offset < i; --i) {
      const auto T_a = Eigen::Map<const Input>{variables[i - 1]}.variable();
      const auto T_b = Eigen::Map<const Input>{variables[i]}.variable();
      const auto w0_i = weights(i - offset, MotionDerivative::VALUE);

      JacobianNM<SU2Tangent> J_R_i_w_ab, J_d_ab_R_ab;
      const auto R_ab = T_a.rotation().groupInverse().groupPlus(T_b.rotation());
      const auto d_ab = R_ab.toTangent(J_d_ab_R_ab.data());
      const auto x_ab = Translation{T_b.translation() - T_a.translation()};
      const auto w_ab = SU2Tangent{w0_i * d_ab};
      const auto R_i = w_ab.toManifold(J_R_i_w_ab.data());
      const auto x_i = Translation{w0_i * x_ab};

      const auto i_R = R.groupInverse().matrix();
      const auto i_R_ab = R_ab.groupInverse().matrix();

      // Update left value Jacobian.
      const auto J_x_a = (i_R * J_R_i_w_ab * w0_i * J_d_ab_R_ab).eval();
      RotationJacobian(Js[MotionDerivative::VALUE], i - 1).noalias() = -J_x_a * i_R_ab;
      TranslationJacobian(Js[MotionDerivative::VALUE], i - 1).noalias() = -w0_i * JacobianNM<Translation>::Identity();

      // Velocity update.
      if constexpr (MotionDerivative::VALUE < TMotionDerivative) {
        const auto i_R_d_ab = (i_R * d_ab).eval();
        const auto i_R_d_ab_x = i_R_d_ab.hat();
        const auto w1_i = weights(i - offset, MotionDerivative::VELOCITY);
        const auto w1_i_i_R_d_ab = (w1_i * i_R_d_ab).eval();

        v.angular() += w1_i_i_R_d_ab;
        v.linear() += w1_i * x_ab;

        const auto J_v_a = (w1_i * i_R * J_d_ab_R_ab).eval();
        const auto J_v_b = (w1_i * i_R_d_ab_x).eval();

        // Update velocity Jacobians.
        RotationJacobian(Js[MotionDerivative::VELOCITY], i - 1).noalias() = -J_v_a * i_R_ab;
        TranslationJacobian(Js[MotionDerivative::VELOCITY], i - 1).noalias() = -w1_i * JacobianNM<Translation>::Identity();
        RotationJacobian(Js[MotionDerivative::VELOCITY], i).noalias() += J_v_a + J_v_b * RotationJacobian(Js[MotionDerivative::VALUE], i);
        TranslationJacobian(Js[MotionDerivative::VELOCITY], i).noalias() += w1_i * JacobianNM<Translation>::Identity();

        // Propagate velocity updates.
        for (Index k = last_idx; i < k; --k) {
          RotationJacobian(Js[MotionDerivative::VELOCITY], k).noalias() += J_v_b * RotationJacobian(Js[MotionDerivative::VALUE], k);
        }

        // Acceleration update.
        if constexpr (MotionDerivative::VELOCITY < TMotionDerivative) {
          const auto w2_i = weights(i - offset, MotionDerivative::ACCELERATION);
          const auto w1_i_i_R_d_ab_x = w1_i_i_R_d_ab.hat();

          a.angular() += w2_i * i_R_d_ab + w1_i_i_R_d_ab.cross(v.angular());
          a.linear() += w2_i * x_ab;

          const auto v_x = v.angular().hat();
          const auto J_a_a = (w2_i * i_R * J_d_ab_R_ab).eval();
          const auto J_a_b = (w2_i * i_R_d_ab_x).eval();
          const auto J_a_c = (J_a_b - v_x * J_v_b).eval();

          // Update acceleration Jacobians.
          RotationJacobian(Js[MotionDerivative::ACCELERATION], i - 1).noalias() = -J_a_a * i_R_ab + (w1_i_i_R_d_ab_x - v_x) * RotationJacobian(Js[MotionDerivative::VELOCITY], i - 1);
          TranslationJacobian(Js[MotionDerivative::ACCELERATION], i - 1).noalias() = -w2_i * JacobianNM<Translation>::Identity();
          RotationJacobian(Js[MotionDerivative::ACCELERATION], i).noalias() += J_a_a + J_a_b * RotationJacobian(Js[MotionDerivative::VALUE], i) + (w1_i_i_R_d_ab_x - v_x) * RotationJacobian(Js[MotionDerivative::VELOCITY], i);
          TranslationJacobian(Js[MotionDerivative::ACCELERATION], i).noalias() += w2_i * JacobianNM<Translation>::Identity();

          // Propagate acceleration updates.
          for (Index k = last_idx; i < k; --k) {
            RotationJacobian(Js[MotionDerivative::ACCELERATION], k).noalias() += J_a_c * RotationJacobian(Js[MotionDerivative::VALUE], k) + w1_i_i_R_d_ab_x * RotationJacobian(Js[MotionDerivative::VELOCITY], k);
          }
        }
      }

      // Update right value Jacobian.
      RotationJacobian(Js[MotionDerivative::VALUE], i).noalias() += J_x_a;
      TranslationJacobian(Js[MotionDerivative::VALUE], i).noalias() += w0_i * JacobianNM<Translation>::Identity();

      // Value update.
      R = R_i * R;
      x = x_i + x;
    }

    const auto T_a = Eigen::Map<const Input>{variables[offset]}.variable();
    RotationJacobian(Js[MotionDerivative::VALUE], offset).noalias() += R.groupInverse().matrix();
    TranslationJacobian(Js[MotionDerivative::VALUE], offset).noalias() += JacobianNM<Translation>::Identity();

    R = T_a.rotation() * R;
    x = T_a.translation() + x;

    for (Index i = offset; i < end_idx; ++i) {
      const auto adapter = SU2JacobianAdapter(variables[i] + Input::kVariableOffset + Manifold::kRotationOffset);
      for (auto& J : Js) {
        RotationJacobian<SU2<Scalar>::kNumParameters>(J, i) = RotationJacobian(J, i) * adapter;
      }
    }
  }

  Eigen::Map<Manifold>{outputs[MotionDerivative::VALUE]} = SE3<Scalar>{R, x};
  if constexpr (MotionDerivative::VALUE < TMotionDerivative) {
    Eigen::Map<Tangent>{outputs[MotionDerivative::VELOCITY]} = v;
    if constexpr (MotionDerivative::VELOCITY < TMotionDerivative) {
      Eigen::Map<Tangent>{outputs[MotionDerivative::ACCELERATION]} = a;
    }
  }

  return true;
}

} // namespace hyper
