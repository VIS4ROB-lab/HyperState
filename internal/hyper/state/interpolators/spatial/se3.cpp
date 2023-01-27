/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include "hyper/state/interpolators/spatial/se3.hpp"
#include "hyper/state/interpolators/spatial/cartesian.hpp"
#include "hyper/variables/groups/adapters.hpp"
#include "hyper/variables/stamped.hpp"

namespace hyper::state {

using namespace variables;

template <typename TScalar>
auto SpatialInterpolator<SE3<TScalar>>::evaluate(const Index& derivative, const Scalar* const* inputs, const Index& num_inputs, const Index& start_index, const Index& end_index,
                                                 const Index& num_input_parameters, const Index& input_offset, const Eigen::Ref<const MatrixX<Scalar>>& weights, bool jacobians)
    -> Result<Output> {
  // Definitions.
  using Rotation = typename Input::Rotation;
  using Translation = typename Input::Translation;
  using Velocity = Tangent<Input>;
  using Acceleration = Tangent<Input>;

  using SU2 = variables::SU2<TScalar>;
  using SU2Tangent = Tangent<SU2>;
  using SU2Jacobian = JacobianNM<SU2Tangent>;
  using SU2Adapter = JacobianNM<SU2Tangent, SU2>;

  // Constants.
  constexpr auto kValue = 0;
  constexpr auto kVelocity = 1;
  constexpr auto kAcceleration = 2;

  // Allocate result.
  auto result = Result<Output>(derivative, jacobians, num_inputs, num_input_parameters);

  // Input lambda definition.
  auto I = [&inputs, &input_offset](const Index& i) {
    return Eigen::Map<const Input>{inputs[i] + input_offset};
  };

  // Allocate accumulators.
  Rotation R = Rotation::Identity();
  Translation x = Translation::Zero();
  Velocity v = Velocity::Zero();
  Acceleration a = Acceleration::Zero();

  if (!jacobians) {
    // Retrieves first input.
    const auto T_0 = I(start_index);

    for (Index i = end_index; start_index < i; --i) {
      const auto T_a = I(i - 1);
      const auto T_b = I(i);
      const auto w0_i = weights(i - start_index, kValue);

      const auto R_ab = T_a.rotation().gInv().gPlus(T_b.rotation());
      const auto d_ab = R_ab.gLog();
      const auto x_ab = Translation{T_b.translation() - T_a.translation()};
      const auto w_ab = SU2Tangent{w0_i * d_ab};
      const auto R_i = w_ab.gExp();
      const auto x_i = Translation{w0_i * x_ab};

      if (kValue < derivative) {
        const auto i_R = R.gInv().gAdj();
        const auto i_R_d_ab = (i_R * d_ab).eval();
        const auto w1_i = weights(i - start_index, kVelocity);
        const auto w1_i_i_R_d_ab = (w1_i * i_R_d_ab).eval();

        v.angular() += w1_i_i_R_d_ab;
        v.linear() += w1_i * x_ab;

        if (kVelocity < derivative) {
          const auto w2_i = weights(i - start_index, kAcceleration);
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
    // Jacobian lambda definitions.
    auto Jr = [&result, &input_offset](const Index& k, const Index& i) {
      using Tangent = Tangent<Input>;
      return result.template jacobian<Tangent::Angular::kNumParameters, SU2Tangent::kNumParameters>(k, i, Tangent::kAngularOffset, Input::kRotationOffset + input_offset);
    };

    auto Jq = [&result, &input_offset](const Index& k, const Index& i) {
      using Tangent = Tangent<Input>;
      return result.template jacobian<Tangent::Angular::kNumParameters, Rotation::kNumParameters>(k, i, Tangent::kAngularOffset, Input::kRotationOffset + input_offset);
    };

    auto Jt = [&result, &input_offset](const Index& k, const Index& i) {
      using Tangent = Tangent<Input>;
      return result.template jacobian<Tangent::Linear::kNumParameters, Translation::kNumParameters>(k, i, Tangent::kLinearOffset, Input::kTranslationOffset + input_offset);
    };

    for (Index i = end_index; start_index < i; --i) {
      const auto T_a = I(i - 1);
      const auto T_b = I(i);
      const auto w0_i = weights(i - start_index, kValue);

      SU2Jacobian J_R_i_w_ab, J_d_ab_R_ab;
      const auto R_ab = T_a.rotation().gInv().gPlus(T_b.rotation());
      const auto d_ab = R_ab.gLog(J_d_ab_R_ab.data());
      const auto x_ab = Translation{T_b.translation() - T_a.translation()};
      const auto w_ab = SU2Tangent{w0_i * d_ab};
      const auto R_i = w_ab.gExp(J_R_i_w_ab.data());
      const auto x_i = Translation{w0_i * x_ab};

      const auto i_R = R.gInv().gAdj();
      const auto i_R_ab = R_ab.gInv().gAdj();

      const auto J_x_0 = (i_R * J_R_i_w_ab * w0_i * J_d_ab_R_ab).eval();

      // Update left value Jacobian.
      Jr(kValue, i - 1).noalias() = -J_x_0 * i_R_ab;
      Jt(kValue, i - 1).diagonal().array() = -w0_i;

      // Velocity update.
      if (kValue < derivative) {
        const auto i_R_d_ab = (i_R * d_ab).eval();
        const auto i_R_d_ab_x = i_R_d_ab.hat();
        const auto w1_i = weights(i - start_index, kVelocity);
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
        for (Index k = end_index; i < k; --k) {
          Jr(kVelocity, k).noalias() += J_v_1 * Jr(kValue, k);
        }

        // Acceleration update.
        if (kVelocity < derivative) {
          const auto w2_i = weights(i - start_index, kAcceleration);
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
          for (Index k = end_index; i < k; --k) {
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

    Jr(kValue, start_index).noalias() += R.gInv().gAdj();
    Jt(kValue, start_index).diagonal().array() += TScalar{1};

    // Apply Jacobian adapters.
    for (Index i = start_index; i <= end_index; ++i) {
      const auto Ja = JacobianAdapter<SU2>(inputs[i] + Input::kRotationOffset);
      for (Index k = 0; k <= derivative; ++k) {
        Jq(k, i) = Jr(k, i) * Ja;
      }
    }

    const auto T_a = I(start_index);
    R = T_a.rotation() * R;
    x = T_a.translation() + x;
  }

  result.value = Output{R, x};
  if (kValue < derivative) {
    result.derivative(kVelocity - 1) = v;
    if (kVelocity < derivative) {
      result.derivative(kAcceleration - 1) = a;
    }
  }

  return result;
}

template <typename TScalar>
auto SpatialInterpolator<SE3<TScalar>, Tangent<SE3<TScalar>>>::evaluate(const Index& derivative, const Scalar* const* inputs, const Index& num_inputs, const Index& start_index,
                                                                        const Index& end_index, const Index& num_input_parameters, const Index& input_offset,
                                                                        const Eigen::Ref<const MatrixX<Scalar>>& weights, bool jacobians) -> Result<Output> {
  // Allocate result.
  auto o_result = Result<Output>(derivative, jacobians, num_inputs, num_input_parameters);
  auto i_result =
      SpatialInterpolator<Tangent<SE3<TScalar>>>::evaluate(derivative, inputs, num_inputs, start_index, end_index, num_input_parameters, input_offset, weights, jacobians);

  if (!jacobians) {
    o_result.value = i_result.value.gExp();
    return o_result;
  } else {
    JacobianNM<Tangent<SE3<TScalar>>> J_m;
    o_result.value = i_result.value.gExp(J_m.data());
    o_result.matrix = J_m * i_result.matrix;
    return o_result;
  }
}

template class SpatialInterpolator<SE3<double>>;
template class SpatialInterpolator<SE3<double>, Tangent<SE3<double>>>;

}  // namespace hyper::state
