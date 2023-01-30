/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include "hyper/state/interpolators/spatial/su2.hpp"
#include "hyper/variables/groups/adapters.hpp"
#include "hyper/variables/stamped.hpp"

namespace hyper::state {

using namespace variables;

template <typename TScalar>
auto SpatialInterpolator<SU2<TScalar>>::evaluate(const Index& derivative, const Scalar* const* inputs, const Index& num_inputs, const Index& start_index, const Index& end_index,
                                                 const Index& num_input_parameters, const Index& input_offset, const Eigen::Ref<const MatrixX<Scalar>>& weights, bool jacobians)
    -> Result<Output> {
  // Definitions.
  using Velocity = variables::Tangent<Output>;
  using Acceleration = variables::Tangent<Output>;

  using Tangent = variables::Tangent<Output>;
  using Jacobian = variables::JacobianNM<Tangent>;

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
  Output R = Output::Identity();
  Velocity v = Velocity::Zero();
  Acceleration a = Acceleration::Zero();

  if (!jacobians) {
    // Retrieves first input.
    const auto I_0 = I(start_index);

    for (Index i = end_index; start_index < i; --i) {
      const auto I_a = I(i - 1);
      const auto I_b = I(i);
      const auto w0_i = weights(i - start_index, kValue);

      const auto R_ab = I_a.gInv().gPlus(I_b);
      const auto d_ab = R_ab.gLog();
      const auto w_ab = Tangent{w0_i * d_ab};
      const auto R_i = w_ab.gExp();

      if (kValue < derivative) {
        const auto i_R = R.gInv().gAdj();
        const auto i_R_d_ab = (i_R * d_ab).eval();
        const auto w1_i = weights(i - start_index, kVelocity);
        const auto w1_i_i_R_d_ab = (w1_i * i_R_d_ab).eval();

        v += w1_i_i_R_d_ab;

        if (kVelocity < derivative) {
          const auto w2_i = weights(i - start_index, kAcceleration);
          a += w2_i * i_R_d_ab + w1_i_i_R_d_ab.cross(v);
        }
      }

      R = R_i * R;
    }

    R = I_0 * R;

  } else {
    // Jacobian lambda definitions.
    auto Jr = [&result, &input_offset](const Index& k, const Index& i) {
      return result.template jacobian<Tangent::kNumParameters, Tangent::kNumParameters>(k, i, Tangent::kAngularOffset, input_offset);
    };

    auto Jq = [&result, &input_offset](const Index& k, const Index& i) {
      return result.template jacobian<Tangent::kNumParameters, Output::kNumParameters>(k, i, Tangent::kAngularOffset, input_offset);
    };

    for (Index i = end_index; start_index < i; --i) {
      const auto I_a = I(i - 1);
      const auto I_b = I(i);
      const auto w0_i = weights(i - start_index, kValue);

      Jacobian J_R_i_w_ab, J_d_ab_R_ab;
      const auto R_ab = I_a.gInv().gPlus(I_b);
      const auto d_ab = R_ab.gLog(J_d_ab_R_ab.data());
      const auto w_ab = Tangent{w0_i * d_ab};
      const auto R_i = w_ab.gExp(J_R_i_w_ab.data());

      const auto i_R = R.gInv().gAdj();
      const auto i_R_ab = R_ab.gInv().gAdj();

      const auto J_x_0 = (i_R * J_R_i_w_ab * w0_i * J_d_ab_R_ab).eval();

      // Update left value Jacobian.
      Jr(kValue, i - 1).noalias() = -J_x_0 * i_R_ab;

      // Velocity update.
      if (kValue < derivative) {
        const auto i_R_d_ab = (i_R * d_ab).eval();
        const auto i_R_d_ab_x = i_R_d_ab.hat();
        const auto w1_i = weights(i - start_index, kVelocity);
        const auto w1_i_i_R_d_ab = (w1_i * i_R_d_ab).eval();

        v += w1_i_i_R_d_ab;

        const auto i_R_J_d_ab_R_ab = (i_R * J_d_ab_R_ab).eval();
        const auto J_v_0 = (w1_i * i_R_J_d_ab_R_ab).eval();
        const auto J_v_1 = (w1_i * i_R_d_ab_x).eval();

        // Update left velocity Jacobians.
        Jr(kVelocity, i - 1).noalias() = -J_v_0 * i_R_ab;

        // Propagate velocity updates.
        for (Index k = end_index; i < k; --k) {
          Jr(kVelocity, k).noalias() += J_v_1 * Jr(kValue, k);
        }

        // Acceleration update.
        if (kVelocity < derivative) {
          const auto w2_i = weights(i - start_index, kAcceleration);
          const auto w1_i_i_R_d_ab_x = w1_i_i_R_d_ab.hat();

          a += w2_i * i_R_d_ab + w1_i_i_R_d_ab.cross(v);

          const auto v_x = v.hat();
          const auto J_a_0 = (w2_i * i_R_J_d_ab_R_ab).eval();
          const auto J_a_1 = (w1_i_i_R_d_ab_x - v_x).eval();
          const auto J_a_2 = (w2_i * i_R_d_ab_x).eval();
          const auto J_a_3 = (J_a_2 - v_x * J_v_1).eval();

          // Update acceleration Jacobians.
          Jr(kAcceleration, i - 1).noalias() = -J_a_0 * i_R_ab + J_a_1 * Jr(kVelocity, i - 1);
          Jr(kAcceleration, i).noalias() += (J_a_0 + J_a_1 * J_v_0) + (J_a_2 + J_a_1 * J_v_1) * Jr(kValue, i) + w1_i_i_R_d_ab_x * Jr(kVelocity, i);

          // Propagate acceleration updates.
          for (Index k = end_index; i < k; --k) {
            Jr(kAcceleration, k).noalias() += J_a_3 * Jr(kValue, k) + w1_i_i_R_d_ab_x * Jr(kVelocity, k);
          }
        }

        // Update right velocity Jacobian.
        Jr(kVelocity, i).noalias() += J_v_0 + J_v_1 * Jr(kValue, i);
      }

      // Update right value Jacobian.
      Jr(kValue, i).noalias() += J_x_0;

      // Value update.
      R = R_i * R;
    }

    Jr(kValue, start_index).noalias() += R.gInv().gAdj();

    // Apply Jacobian adapters.
    for (Index i = start_index; i <= end_index; ++i) {
      const auto Ja = JacobianAdapter<Output>(inputs[i]);
      for (Index k = 0; k <= derivative; ++k) {
        Jq(k, i) = Jr(k, i) * Ja;
      }
    }

    const auto I_a = I(start_index);
    R = I_a.gPlus(R);
  }

  result.value = R;
  if (kValue < derivative) {
    result.derivative(kVelocity - 1) = v;
    if (kVelocity < derivative) {
      result.derivative(kAcceleration - 1) = a;
    }
  }

  return result;
}

template class SpatialInterpolator<SU2<double>>;
//template class SpatialInterpolator<SU2<double>, Tangent<SU2<double>>>;

}  // namespace hyper::state
