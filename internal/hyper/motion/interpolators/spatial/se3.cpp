/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include "hyper/motion/interpolators/spatial/se3.hpp"
#include "hyper/variables/adapters.hpp"
#include "hyper/variables/groups/se3.hpp"
#include "hyper/variables/stamped.hpp"

namespace hyper {

namespace {

using Value = SE3<Scalar>;
using Input = Stamped<Value>;
using Derivative = Tangent<Value>;
using SE3Tangent = Tangent<SE3<Scalar>>;
using SU2Tangent = Tangent<SU2<Scalar>>;

constexpr auto kNumValueParameters = Value::kNumParameters;
constexpr auto kNumInputParameters = Input::kNumParameters;
constexpr auto kNumDerivativeParameters = Derivative::kNumParameters;

constexpr auto kValueIndex = 0;
constexpr auto kVelocityIndex = 1;
constexpr auto kAccelerationIndex = 2;

template <Index TCols = 3, typename TMatrix>
inline auto RotationJacobian(TMatrix& matrix, const Index& index) {
  return matrix.template block<3, TCols>(0, index * kNumInputParameters + 0);
}

template <typename TMatrix>
inline auto TranslationJacobian(TMatrix& matrix, const Index& index) {
  return matrix.template block<3, 3>(3, index * kNumInputParameters + 4);
}

} // namespace

auto SpatialInterpolator<Stamped<SE3<Scalar>>>::evaluate(const StateQuery& state_query, const SpatialQuery& spatial_query) -> StateResult {
  const auto derivative = state_query.derivative;
  switch (derivative) {
    case kValueIndex:
      return evaluate<kValueIndex>(state_query, spatial_query);
    case kVelocityIndex:
      return evaluate<kVelocityIndex>(state_query, spatial_query);
    case kAccelerationIndex:
      return evaluate<kAccelerationIndex>(state_query, spatial_query);
    default:
      LOG(FATAL) << "Requested derivative is not available.";
      return {};
  }
}

template <int TDerivative>
auto SpatialInterpolator<Stamped<SE3<Scalar>>>::evaluate(const StateQuery& state_query, const SpatialQuery& policy_query) -> StateResult {
  // Definitions.
  using Result = StateResult;
  using Rotation = typename SE3<Scalar>::Rotation;
  using Translation = typename SE3<Scalar>::Translation;

  // Unpack queries.
  const auto& [stamp, _, jacobian] = state_query;
  const auto& [layout, inputs, weights] = policy_query;

  // Sanity checks.
  DCHECK(weights.rows() == layout.inner_input_size && weights.cols() == TDerivative + 1);

  // Allocate result.
  Result result;
  auto& [outputs, jacobians] = result;
  outputs.reserve(TDerivative + 1);
  jacobians.reserve(TDerivative + 1);

  // Compute indices.
  const auto start_idx = layout.left_input_padding;
  const auto end_idx = start_idx + layout.inner_input_size - 1;

  // Allocate accumulators.
  Rotation R = Rotation::Identity();
  Translation x = Translation::Zero();
  SE3Tangent v = SE3Tangent::Zero();
  SE3Tangent a = SE3Tangent::Zero();

  if (!jacobian) {
    // Retrieves first input.
    const auto T_0 = Eigen::Map<const Input>{inputs[start_idx]}.variable();

    /* Allocate accumulators.
    Rotation R = T_0.rotation();
    Translation x = T_0.translation();
    SE3Tangent v = SE3Tangent::Zero();
    SE3Tangent a = SE3Tangent::Zero();

    for (Index i = start_idx; i < end_idx; ++i) {
      const auto T_a = Eigen::Map<const Input>{inputs[i]}.variable();
      const auto T_b = Eigen::Map<const Input>{inputs[i + 1]}.variable();
      const auto w0_i = weights(i - start_idx + 1, kValueIndex);

      const auto R_ab = T_a.rotation().groupInverse().groupPlus(T_b.rotation());
      const auto d_ab = R_ab.toTangent();
      const auto x_ab = Translation{T_b.translation() - T_a.translation()};
      const auto w_ab = SU2Tangent{w0_i * d_ab};
      const auto R_i = w_ab.toManifold();
      const auto x_i = Translation{w0_i * x_ab};

      R *= R_i;
      x += x_i;

      if constexpr (kValueIndex < TDerivative) {
        const auto i_R_i = R_i.groupInverse();
        const auto w1_i = weights(i - start_idx + 1, kVelocityIndex);
        v.angular() = i_R_i * v.angular() + w1_i * d_ab;
        v.linear() += w1_i * x_ab;

        if constexpr (kVelocityIndex < TDerivative) {
          const auto w2_i = weights(i - start_idx + 1, kAccelerationIndex);
          a.angular() = i_R_i * a.angular() + w1_i * v.angular().cross(d_ab) + w2_i * d_ab;
          a.linear() += w2_i * x_ab;
        }
      }
    } */

    for (Index i = end_idx; start_idx < i; --i) {
      const auto T_a = Eigen::Map<const Input>{inputs[i - 1]}.variable();
      const auto T_b = Eigen::Map<const Input>{inputs[i]}.variable();
      const auto w0_i = weights(i - start_idx, kValueIndex);

      const auto R_ab = T_a.rotation().groupInverse().groupPlus(T_b.rotation());
      const auto d_ab = R_ab.toTangent();
      const auto x_ab = Translation{T_b.translation() - T_a.translation()};
      const auto w_ab = SU2Tangent{w0_i * d_ab};
      const auto R_i = w_ab.toManifold();
      const auto x_i = Translation{w0_i * x_ab};

      if constexpr (kValueIndex < TDerivative) {
        const auto i_R = R.groupInverse().matrix();
        const auto i_R_d_ab = (i_R * d_ab).eval();
        const auto w1_i = weights(i - start_idx, kVelocityIndex);
        const auto w1_i_i_R_d_ab = (w1_i * i_R_d_ab).eval();
        v.angular() += w1_i_i_R_d_ab;
        v.linear() += w1_i * x_ab;

        if constexpr (kVelocityIndex < TDerivative) {
          const auto w2_i = weights(i - start_idx, kAccelerationIndex);
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
    const auto num_parameters = layout.outer_input_size * kNumInputParameters;
    jacobians.emplace_back(Result::Jacobian::Zero(kNumDerivativeParameters, num_parameters));
    if constexpr (kValueIndex < TDerivative) {
      jacobians.emplace_back(Result::Jacobian::Zero(kNumDerivativeParameters, num_parameters));
      if constexpr (kVelocityIndex < TDerivative) {
        jacobians.emplace_back(Result::Jacobian::Zero(kNumDerivativeParameters, num_parameters));
      }
    }

    for (Index i = end_idx; start_idx < i; --i) {
      const auto T_a = Eigen::Map<const Input>{inputs[i - 1]}.variable();
      const auto T_b = Eigen::Map<const Input>{inputs[i]}.variable();
      const auto w0_i = weights(i - start_idx, kValueIndex);

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
      RotationJacobian(jacobians[kValueIndex], i - 1).noalias() = -J_x_a * i_R_ab;
      TranslationJacobian(jacobians[kValueIndex], i - 1).noalias() = -w0_i * JacobianNM<Translation>::Identity();

      // Velocity update.
      if constexpr (kValueIndex < TDerivative) {
        const auto i_R_d_ab = (i_R * d_ab).eval();
        const auto i_R_d_ab_x = i_R_d_ab.hat();
        const auto w1_i = weights(i - start_idx, kVelocityIndex);
        const auto w1_i_i_R_d_ab = (w1_i * i_R_d_ab).eval();

        v.angular() += w1_i_i_R_d_ab;
        v.linear() += w1_i * x_ab;

        const auto J_v_a = (w1_i * i_R * J_d_ab_R_ab).eval();
        const auto J_v_b = (w1_i * i_R_d_ab_x).eval();

        // Update velocity Jacobians.
        RotationJacobian(jacobians[kVelocityIndex], i - 1).noalias() = -J_v_a * i_R_ab;
        TranslationJacobian(jacobians[kVelocityIndex], i - 1).noalias() = -w1_i * JacobianNM<Translation>::Identity();
        RotationJacobian(jacobians[kVelocityIndex], i).noalias() += J_v_a + J_v_b * RotationJacobian(jacobians[kValueIndex], i);
        TranslationJacobian(jacobians[kVelocityIndex], i).noalias() += w1_i * JacobianNM<Translation>::Identity();

        // Propagate velocity updates.
        for (Index k = end_idx; i < k; --k) {
          RotationJacobian(jacobians[kVelocityIndex], k).noalias() += J_v_b * RotationJacobian(jacobians[kValueIndex], k);
        }

        // Acceleration update.
        if constexpr (kVelocityIndex < TDerivative) {
          const auto w2_i = weights(i - start_idx, kAccelerationIndex);
          const auto w1_i_i_R_d_ab_x = w1_i_i_R_d_ab.hat();

          a.angular() += w2_i * i_R_d_ab + w1_i_i_R_d_ab.cross(v.angular());
          a.linear() += w2_i * x_ab;

          const auto v_x = v.angular().hat();
          const auto J_a_a = (w2_i * i_R * J_d_ab_R_ab).eval();
          const auto J_a_b = (w2_i * i_R_d_ab_x).eval();
          const auto J_a_c = (J_a_b - v_x * J_v_b).eval();

          // Update acceleration Jacobians.
          RotationJacobian(jacobians[kAccelerationIndex], i - 1).noalias() = -J_a_a * i_R_ab + (w1_i_i_R_d_ab_x - v_x) * RotationJacobian(jacobians[kVelocityIndex], i - 1);
          TranslationJacobian(jacobians[kAccelerationIndex], i - 1).noalias() = -w2_i * JacobianNM<Translation>::Identity();
          RotationJacobian(jacobians[kAccelerationIndex], i).noalias() += J_a_a + J_a_b * RotationJacobian(jacobians[kValueIndex], i) + (w1_i_i_R_d_ab_x - v_x) * RotationJacobian(jacobians[kVelocityIndex], i);
          TranslationJacobian(jacobians[kAccelerationIndex], i).noalias() += w2_i * JacobianNM<Translation>::Identity();

          // Propagate acceleration updates.
          for (Index k = end_idx; i < k; --k) {
            RotationJacobian(jacobians[kAccelerationIndex], k).noalias() += J_a_c * RotationJacobian(jacobians[kValueIndex], k) + w1_i_i_R_d_ab_x * RotationJacobian(jacobians[kVelocityIndex], k);
          }
        }
      }

      // Update right value Jacobian.
      RotationJacobian(jacobians[kValueIndex], i).noalias() += J_x_a;
      TranslationJacobian(jacobians[kValueIndex], i).noalias() += w0_i * JacobianNM<Translation>::Identity();

      // Value update.
      R = R_i * R;
      x = x_i + x;
    }

    const auto T_a = Eigen::Map<const Input>{inputs[start_idx]}.variable();
    RotationJacobian(jacobians[kValueIndex], start_idx).noalias() += R.groupInverse().matrix();
    TranslationJacobian(jacobians[kValueIndex], start_idx).noalias() += JacobianNM<Translation>::Identity();

    R = T_a.rotation() * R;
    x = T_a.translation() + x;

    for (Index i = start_idx; i <= end_idx; ++i) {
      const auto adapter = SU2JacobianAdapter(inputs[i] + Input::kVariableOffset + Value::kRotationOffset);
      for (auto& J : jacobians) {
        RotationJacobian<SU2<Scalar>::kNumParameters>(J, i) = RotationJacobian(J, i) * adapter;
      }
    }
  }

  outputs.emplace_back(SE3<Scalar>{R, x});
  if constexpr (kValueIndex < TDerivative) {
    outputs.emplace_back(v);
    if constexpr (kVelocityIndex < TDerivative) {
      outputs.emplace_back(a);
    }
  }

  return result;
}

} // namespace hyper
