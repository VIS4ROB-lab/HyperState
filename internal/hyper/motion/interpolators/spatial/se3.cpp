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

auto SpatialInterpolator<Stamped<SE3<Scalar>>>::evaluate(
    const TemporalMotionQuery<Scalar>& query,
    const TemporalInterpolatorLayout<Index>& layout,
    const Eigen::Ref<const MatrixX<Scalar>>& weights,
    const Scalar* const* inputs) -> TemporalMotionResult<Scalar> {
  switch (query.derivative) {
    case MotionDerivative::VALUE:
      return evaluate<MotionDerivative::VALUE>(query, layout, weights, inputs);
    case MotionDerivative::VELOCITY:
      return evaluate<MotionDerivative::VELOCITY>(query, layout, weights, inputs);
    case MotionDerivative::ACCELERATION:
      return evaluate<MotionDerivative::ACCELERATION>(query, layout, weights, inputs);
    default:
      LOG(FATAL) << "Requested derivative is not available.";
      return {};
  }
}

template <MotionDerivative TMotionDerivative>
auto SpatialInterpolator<Stamped<SE3<Scalar>>>::evaluate(
    const TemporalMotionQuery<Scalar>& query,
    const TemporalInterpolatorLayout<Index>& layout,
    const Eigen::Ref<const MatrixX<Scalar>>& weights,
    const Scalar* const* inputs) -> TemporalMotionResult<Scalar> {
  // Definitions.
  using Rotation = typename SE3<Scalar>::Rotation;
  using Translation = typename SE3<Scalar>::Translation;
  using SU2Tangent = hyper::Tangent<SU2<Scalar>>;

  // Unpack queries.
  const auto& [time, derivative, jacobian] = query;

  // Sanity checks.
  DCHECK(weights.rows() == layout.inner_input_size && weights.cols() == TMotionDerivative + 1);

  // Allocate result.
  TemporalMotionResult<Scalar> result;
  auto& [derivatives, jacobians] = result;
  derivatives.reserve(TMotionDerivative + 1);
  jacobians.reserve(TMotionDerivative + 1);

  // Compute indices.
  const auto start_idx = layout.left_input_padding;
  const auto end_idx = start_idx + layout.inner_input_size - 1;

  // Allocate accumulators.
  Rotation R = Rotation::Identity();
  Translation x = Translation::Zero();
  Tangent v = Tangent::Zero();
  Tangent a = Tangent::Zero();

  if (!jacobian) {
    // Retrieves first input.
    const auto T_0 = Eigen::Map<const Input>{inputs[start_idx]}.variable();

    /* Allocate accumulators.
    Rotation R = T_0.rotation();
    Translation x = T_0.translation();
    Tangent v = Tangent::Zero();
    Tangent a = Tangent::Zero();

    for (Index i = start_idx; i < end_idx; ++i) {
      const auto T_a = Eigen::Map<const Input>{inputs[i]}.variable();
      const auto T_b = Eigen::Map<const Input>{inputs[i + 1]}.variable();
      const auto w0_i = weights(i - start_idx + 1, Derivative::VALUE);

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
        const auto w1_i = weights(i - start_idx + 1, Derivative::VELOCITY);
        v.angular() = i_R_i * v.angular() + w1_i * d_ab;
        v.linear() += w1_i * x_ab;

        if constexpr (Derivative::VELOCITY < TMotionDerivative) {
          const auto w2_i = weights(i - start_idx + 1, Derivative::ACCELERATION);
          a.angular() = i_R_i * a.angular() + w1_i * v.angular().cross(d_ab) + w2_i * d_ab;
          a.linear() += w2_i * x_ab;
        }
      }
    } */

    for (Index i = end_idx; start_idx < i; --i) {
      const auto T_a = Eigen::Map<const Input>{inputs[i - 1]}.variable();
      const auto T_b = Eigen::Map<const Input>{inputs[i]}.variable();
      const auto w0_i = weights(i - start_idx, MotionDerivative::VALUE);

      const auto R_ab = T_a.rotation().groupInverse().groupPlus(T_b.rotation());
      const auto d_ab = R_ab.toTangent();
      const auto x_ab = Translation{T_b.translation() - T_a.translation()};
      const auto w_ab = SU2Tangent{w0_i * d_ab};
      const auto R_i = w_ab.toManifold();
      const auto x_i = Translation{w0_i * x_ab};

      if constexpr (MotionDerivative::VALUE < TMotionDerivative) {
        const auto i_R = R.groupInverse().matrix();
        const auto i_R_d_ab = (i_R * d_ab).eval();
        const auto w1_i = weights(i - start_idx, MotionDerivative::VELOCITY);
        const auto w1_i_i_R_d_ab = (w1_i * i_R_d_ab).eval();
        v.angular() += w1_i_i_R_d_ab;
        v.linear() += w1_i * x_ab;

        if constexpr (MotionDerivative::VELOCITY < TMotionDerivative) {
          const auto w2_i = weights(i - start_idx, MotionDerivative::ACCELERATION);
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
    jacobians.emplace_back(JacobianX<Scalar>::Zero(kDimTangent, num_parameters));
    if constexpr (MotionDerivative::VALUE < TMotionDerivative) {
      jacobians.emplace_back(JacobianX<Scalar>::Zero(kDimTangent, num_parameters));
      if constexpr (MotionDerivative::VELOCITY < TMotionDerivative) {
        jacobians.emplace_back(JacobianX<Scalar>::Zero(kDimTangent, num_parameters));
      }
    }

    for (Index i = end_idx; start_idx < i; --i) {
      const auto T_a = Eigen::Map<const Input>{inputs[i - 1]}.variable();
      const auto T_b = Eigen::Map<const Input>{inputs[i]}.variable();
      const auto w0_i = weights(i - start_idx, MotionDerivative::VALUE);

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
      RotationJacobian(jacobians[MotionDerivative::VALUE], i - 1).noalias() = -J_x_a * i_R_ab;
      TranslationJacobian(jacobians[MotionDerivative::VALUE], i - 1).noalias() = -w0_i * JacobianNM<Translation>::Identity();

      // Velocity update.
      if constexpr (MotionDerivative::VALUE < TMotionDerivative) {
        const auto i_R_d_ab = (i_R * d_ab).eval();
        const auto i_R_d_ab_x = i_R_d_ab.hat();
        const auto w1_i = weights(i - start_idx, MotionDerivative::VELOCITY);
        const auto w1_i_i_R_d_ab = (w1_i * i_R_d_ab).eval();

        v.angular() += w1_i_i_R_d_ab;
        v.linear() += w1_i * x_ab;

        const auto J_v_a = (w1_i * i_R * J_d_ab_R_ab).eval();
        const auto J_v_b = (w1_i * i_R_d_ab_x).eval();

        // Update velocity Jacobians.
        RotationJacobian(jacobians[MotionDerivative::VELOCITY], i - 1).noalias() = -J_v_a * i_R_ab;
        TranslationJacobian(jacobians[MotionDerivative::VELOCITY], i - 1).noalias() = -w1_i * JacobianNM<Translation>::Identity();
        RotationJacobian(jacobians[MotionDerivative::VELOCITY], i).noalias() += J_v_a + J_v_b * RotationJacobian(jacobians[MotionDerivative::VALUE], i);
        TranslationJacobian(jacobians[MotionDerivative::VELOCITY], i).noalias() += w1_i * JacobianNM<Translation>::Identity();

        // Propagate velocity updates.
        for (Index k = end_idx; i < k; --k) {
          RotationJacobian(jacobians[MotionDerivative::VELOCITY], k).noalias() += J_v_b * RotationJacobian(jacobians[MotionDerivative::VALUE], k);
        }

        // Acceleration update.
        if constexpr (MotionDerivative::VELOCITY < TMotionDerivative) {
          const auto w2_i = weights(i - start_idx, MotionDerivative::ACCELERATION);
          const auto w1_i_i_R_d_ab_x = w1_i_i_R_d_ab.hat();

          a.angular() += w2_i * i_R_d_ab + w1_i_i_R_d_ab.cross(v.angular());
          a.linear() += w2_i * x_ab;

          const auto v_x = v.angular().hat();
          const auto J_a_a = (w2_i * i_R * J_d_ab_R_ab).eval();
          const auto J_a_b = (w2_i * i_R_d_ab_x).eval();
          const auto J_a_c = (J_a_b - v_x * J_v_b).eval();

          // Update acceleration Jacobians.
          RotationJacobian(jacobians[MotionDerivative::ACCELERATION], i - 1).noalias() = -J_a_a * i_R_ab + (w1_i_i_R_d_ab_x - v_x) * RotationJacobian(jacobians[MotionDerivative::VELOCITY], i - 1);
          TranslationJacobian(jacobians[MotionDerivative::ACCELERATION], i - 1).noalias() = -w2_i * JacobianNM<Translation>::Identity();
          RotationJacobian(jacobians[MotionDerivative::ACCELERATION], i).noalias() += J_a_a + J_a_b * RotationJacobian(jacobians[MotionDerivative::VALUE], i) + (w1_i_i_R_d_ab_x - v_x) * RotationJacobian(jacobians[MotionDerivative::VELOCITY], i);
          TranslationJacobian(jacobians[MotionDerivative::ACCELERATION], i).noalias() += w2_i * JacobianNM<Translation>::Identity();

          // Propagate acceleration updates.
          for (Index k = end_idx; i < k; --k) {
            RotationJacobian(jacobians[MotionDerivative::ACCELERATION], k).noalias() += J_a_c * RotationJacobian(jacobians[MotionDerivative::VALUE], k) + w1_i_i_R_d_ab_x * RotationJacobian(jacobians[MotionDerivative::VELOCITY], k);
          }
        }
      }

      // Update right value Jacobian.
      RotationJacobian(jacobians[MotionDerivative::VALUE], i).noalias() += J_x_a;
      TranslationJacobian(jacobians[MotionDerivative::VALUE], i).noalias() += w0_i * JacobianNM<Translation>::Identity();

      // Value update.
      R = R_i * R;
      x = x_i + x;
    }

    const auto T_a = Eigen::Map<const Input>{inputs[start_idx]}.variable();
    RotationJacobian(jacobians[MotionDerivative::VALUE], start_idx).noalias() += R.groupInverse().matrix();
    TranslationJacobian(jacobians[MotionDerivative::VALUE], start_idx).noalias() += JacobianNM<Translation>::Identity();

    R = T_a.rotation() * R;
    x = T_a.translation() + x;

    for (Index i = start_idx; i <= end_idx; ++i) {
      const auto adapter = SU2JacobianAdapter(inputs[i] + Input::kVariableOffset + Manifold::kRotationOffset);
      for (auto& J : jacobians) {
        RotationJacobian<SU2<Scalar>::kNumParameters>(J, i) = RotationJacobian(J, i) * adapter;
      }
    }
  }

  derivatives.emplace_back(SE3<Scalar>{R, x});
  if constexpr (MotionDerivative::VALUE < TMotionDerivative) {
    derivatives.emplace_back(v);
    if constexpr (MotionDerivative::VELOCITY < TMotionDerivative) {
      derivatives.emplace_back(a);
    }
  }

  return result;
}

} // namespace hyper
